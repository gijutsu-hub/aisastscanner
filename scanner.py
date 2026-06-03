# ╔══════════════════════════════════════════════════════════════╗
# ║  AI-SAST Smart Scanner v2.1 — Google Colab                  ║
# ║  • Auto-detects uploaded model filenames (no path errors)   ║
# ║  • AST call graph — follows function calls across codebase  ║
# ║  • Taint propagation: source → propagate → sink tracking    ║
# ║  • Focal loss model (v4) or standard model (v3) supported   ║
# ║  • Full text report with call-chain evidence                 ║
# ╚══════════════════════════════════════════════════════════════╝

# ═════════════════════════════════════════════════════════════════
# CELL 1 — Silence TF noise & imports
# ═════════════════════════════════════════════════════════════════
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_MODULE_LOADING"]   = "LAZY"
os.environ["TF_XLA_FLAGS"]         = "--tf_xla_enable_xla_devices=false"

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore")

import re, ast, json, pickle, shutil, zipfile
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(f"TensorFlow : {tf.__version__}")
print(f"GPU        : {tf.config.list_physical_devices('GPU')}")
print("✓ Imports done")


# ═════════════════════════════════════════════════════════════════
# CELL 1b — Install graph visualisation deps
# ═════════════════════════════════════════════════════════════════
import subprocess
subprocess.run(["pip", "install", "-q", "networkx", "matplotlib"], check=False)

import networkx as nx
import matplotlib
matplotlib.use("Agg")   # headless — Colab renders via plt.show()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
print("✓ networkx + matplotlib ready")


# ═════════════════════════════════════════════════════════════════
# CELL 2 — Upload model artefacts
#   Uploads to /content/ — we auto-detect the exact filenames
#   so renamed/numbered files (e.g. "vuln_model_v4 (1).keras")
#   still work.
# ═════════════════════════════════════════════════════════════════
from google.colab import files as colab_files

print("Upload your 3 trained model artefacts:")
print("  → vuln_model_v4.keras    (or v3)")
print("  → tokenizer_v4.pkl       (or v3)")
print("  → label_encoder_v4.pkl   (or v3)\n")

uploaded = colab_files.upload()
print(f"\nUploaded files: {list(uploaded.keys())}")

# ── Auto-detect exact paths ───────────────────────────────────────
def find_in_content(*keywords):
    """
    Search /content for a file whose name contains ALL keywords (case-insensitive).
    Returns the Path if found, else None.
    """
    for fname in os.listdir("/content"):
        flower = fname.lower()
        if all(k.lower() in flower for k in keywords):
            p = Path("/content") / fname
            if p.is_file():
                return p
    return None

# Try v4 first, fall back to v3
MODEL_PATH = (find_in_content("vuln_model") or
              find_in_content("model", ".keras"))
TOK_PATH   = (find_in_content("tokenizer") or
              find_in_content("tok"))
ENC_PATH   = (find_in_content("label_encoder") or
              find_in_content("encoder"))

# Derive max sequence length from model version
MAX_LEN = 128 if (MODEL_PATH and "v4" in str(MODEL_PATH).lower()) else 32

print(f"\n{'─'*50}")
print(f"  MODEL_PATH : {MODEL_PATH}")
print(f"  TOK_PATH   : {TOK_PATH}")
print(f"  ENC_PATH   : {ENC_PATH}")
print(f"  MAX_LEN    : {MAX_LEN}")
print(f"{'─'*50}")

# Validate all found
_missing = [name for name, p in
            [("model", MODEL_PATH), ("tokenizer", TOK_PATH), ("encoder", ENC_PATH)]
            if p is None or not p.exists()]
if _missing:
    print(f"\n[ERROR] Could not find: {_missing}")
    print("List of all files in /content:")
    for f in sorted(os.listdir("/content")):
        sz = Path(f"/content/{f}").stat().st_size if Path(f"/content/{f}").is_file() else 0
        print(f"  {f:<45} {sz:,} bytes")
    raise FileNotFoundError(f"Missing artefacts: {_missing}. Re-upload in this cell.")
else:
    print("✓ All artefacts found")


# ═════════════════════════════════════════════════════════════════
# CELL 3 — Upload project ZIP
#   ZIP your codebase first:
#     Mac/Linux : zip -r myproject.zip ./myproject/
#     Windows   : right-click folder → Send to → Compressed
# ═════════════════════════════════════════════════════════════════
print("Upload your project as a ZIP file:")
uploaded_zip = colab_files.upload()
zip_name     = list(uploaded_zip.keys())[0]

EXTRACT_DIR = Path("/content/scan_target")
if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)
EXTRACT_DIR.mkdir()

with zipfile.ZipFile(zip_name, "r") as zf:
    zf.extractall(EXTRACT_DIR)

# Count source files
LANG_MAP = {
    ".py":"python",  ".js":"javascript", ".ts":"javascript",
    ".jsx":"javascript", ".tsx":"javascript", ".java":"java",
    ".php":"php",    ".c":"c",           ".cpp":"c",
    ".h":"c",        ".go":"go",         ".cs":"csharp",
    ".rb":"ruby",
}
SKIP_DIRS = {
    ".git","__pycache__","node_modules",".venv","venv","env",
    "dist","build",".idea",".vscode","vendor","target","bin","obj",
}

_src_files = []
for root, dirs, fnames in os.walk(EXTRACT_DIR):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for fname in fnames:
        if Path(fname).suffix.lower() in LANG_MAP:
            _src_files.append(Path(root) / fname)

print(f"\n✓ Extracted  : {zip_name}")
print(f"✓ Source files: {len(_src_files)}")
print("\nPreview (first 20):")
for fp in sorted(_src_files)[:20]:
    print(f"  {fp.relative_to(EXTRACT_DIR)}")
if len(_src_files) > 20:
    print(f"  ... and {len(_src_files)-20} more")


# ═════════════════════════════════════════════════════════════════
# CELL 4 — Scan config  ← EDIT HERE
# ═════════════════════════════════════════════════════════════════
SCAN_CONFIG = {
    "target"       : EXTRACT_DIR,

    # Confidence threshold (0.0–1.0)
    # 0.50 = balanced  |  0.60 = fewer FPs  |  0.40 = catch more
    "threshold"    : 0.55,

    # Cross-function call depth for Python
    # 1 = direct callers  |  2 = callers of callers
    "call_depth"   : 2,

    # Include SAFE predictions in report?
    "show_safe"    : False,

    # Output file
    "report_file"  : "sast_report.txt",
    "auto_download": True,
}

print(f"✓ Config set")
print(f"  Threshold  : {SCAN_CONFIG['threshold']}")
print(f"  Call depth : {SCAN_CONFIG['call_depth']}")


# ═════════════════════════════════════════════════════════════════
# CELL 5 — Label metadata
# ═════════════════════════════════════════════════════════════════
LABEL_META = {
    "SQLI"                    :("CRITICAL","CWE-89",  "A03:2021 – Injection"),
    "XSS"                     :("HIGH",    "CWE-79",  "A03:2021 – Injection"),
    "COMMAND_INJECTION"       :("CRITICAL","CWE-78",  "A03:2021 – Injection"),
    "RCE"                     :("CRITICAL","CWE-94",  "A03:2021 – Injection"),
    "BUFFER_OVERFLOW"         :("HIGH",    "CWE-120", "A06:2021 – Vulnerable Components"),
    "FORMAT_STRING"           :("HIGH",    "CWE-134", "A03:2021 – Injection"),
    "PATH_TRAVERSAL"          :("HIGH",    "CWE-22",  "A01:2021 – Broken Access Control"),
    "XXE"                     :("HIGH",    "CWE-611", "A05:2021 – Security Misconfiguration"),
    "DESERIALIZATION"         :("CRITICAL","CWE-502", "A08:2021 – Insecure Deserialization"),
    "HARDCODED_SECRET"        :("CRITICAL","CWE-798", "A07:2021 – Identification & Auth Failures"),
    "WEAK_CRYPTO"             :("MEDIUM",  "CWE-326", "A02:2021 – Cryptographic Failures"),
    "SSL_BYPASS"              :("HIGH",    "CWE-295", "A02:2021 – Cryptographic Failures"),
    "OPEN_REDIRECT"           :("MEDIUM",  "CWE-601", "A01:2021 – Broken Access Control"),
    "SSRF"                    :("HIGH",    "CWE-918", "A10:2021 – SSRF"),
    "LDAP_INJECTION"          :("HIGH",    "CWE-90",  "A03:2021 – Injection"),
    "XPATH_INJECTION"         :("HIGH",    "CWE-643", "A03:2021 – Injection"),
    "SSTI"                    :("CRITICAL","CWE-1336","A03:2021 – Injection"),
    "LOG_INJECTION"           :("MEDIUM",  "CWE-117", "A09:2021 – Security Logging Failures"),
    "IDOR"                    :("HIGH",    "CWE-639", "A01:2021 – Broken Access Control"),
    "CSRF"                    :("MEDIUM",  "CWE-352", "A01:2021 – Broken Access Control"),
    "INSECURE_FILE_UPLOAD"    :("HIGH",    "CWE-434", "A04:2021 – Insecure Design"),
    "INTEGER_OVERFLOW"        :("HIGH",    "CWE-190", "A06:2021 – Vulnerable Components"),
    "USE_AFTER_FREE"          :("HIGH",    "CWE-416", "A06:2021 – Vulnerable Components"),
    "NULL_DEREFERENCE"        :("MEDIUM",  "CWE-476", "A06:2021 – Vulnerable Components"),
    "RACE_CONDITION"          :("HIGH",    "CWE-362", "A04:2021 – Insecure Design"),
    "MASS_ASSIGNMENT"         :("HIGH",    "CWE-915", "A04:2021 – Insecure Design"),
    "JWT_WEAKNESS"            :("CRITICAL","CWE-347", "A07:2021 – Identification & Auth Failures"),
    "CORS_MISCONFIGURATION"   :("MEDIUM",  "CWE-942", "A05:2021 – Security Misconfiguration"),
    "REGEX_DOS"               :("MEDIUM",  "CWE-1333","A04:2021 – Insecure Design"),
    "HTTP_RESPONSE_SPLITTING" :("MEDIUM",  "CWE-113", "A03:2021 – Injection"),
    "SESSION_FIXATION"        :("HIGH",    "CWE-384", "A07:2021 – Identification & Auth Failures"),
    "PRIVILEGE_ESCALATION"    :("CRITICAL","CWE-269", "A01:2021 – Broken Access Control"),
    "FILE_INCLUSION"          :("CRITICAL","CWE-98",  "A03:2021 – Injection"),
    "SENSITIVE_DATA_EXPOSURE" :("HIGH",    "CWE-312", "A02:2021 – Cryptographic Failures"),
    "SAFE"                    :("NONE",    "N/A",     "N/A"),
}
SEV_ORDER = {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"NONE":3,"UNKNOWN":4}
print("✓ Metadata loaded")


# ═════════════════════════════════════════════════════════════════
# CELL 6 — Load model
#   compile=False skips deserializing the custom focal_loss function
#   used during training. For inference only weights matter.
# ═════════════════════════════════════════════════════════════════
print(f"\n  Loading model from : {MODEL_PATH}")
print("  This may take 10–30s ...", end=" ", flush=True)

try:
    # Normal load — works for v3 (standard cross-entropy)
    MODEL = tf.keras.models.load_model(str(MODEL_PATH))
    print("✓  (standard load)")

except (TypeError, Exception) as _e:
    if "focal_loss" in str(_e) or "Could not locate" in str(_e):
        # v4 model was trained with custom focal loss — load weights only
        print(f"\n  [INFO] Custom loss detected — reloading with compile=False ...")
        MODEL = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False,       # ← skips focal_loss deserialisation
        )
        # Recompile with dummy loss — has zero effect on model.predict()
        MODEL.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        print("✓  (inference mode — compile=False)")
    else:
        raise

print(f"  Loading tokenizer  : {TOK_PATH} ...", end=" ", flush=True)
with open(TOK_PATH, "rb") as f:
    TOKENIZER = pickle.load(f)
print("✓")

print(f"  Loading encoder    : {ENC_PATH} ...", end=" ", flush=True)
with open(ENC_PATH, "rb") as f:
    ENCODER = pickle.load(f)
print("✓")

print(f"\n  Model input shape  : {MODEL.input_shape}")
print(f"  Output classes     : {len(ENCODER.classes_)}")
print(f"  MAX_LEN            : {MAX_LEN}")
print("\n✓ Model ready")


# ═════════════════════════════════════════════════════════════════
# CELL 7 — Pre-processing & batch predictor
# ═════════════════════════════════════════════════════════════════
def clean_code(s: str) -> str:
    """
    Normalise code snippet for the tokeniser.
    Preserves <NL> token so the model understands function structure
    (v4 model was trained with this; v3 silently ignores it).
    """
    s = s.replace("\r\n", " <NL> ").replace("\n", " <NL> ")
    s = s.lower().strip()
    s = re.sub(r"([(){}\[\];:,.<>+\-*/=!&|^~%@#$])", r" \1 ", s)
    return re.sub(r"\s+", " ", s)


def predict_batch(snippets: list, threshold: float) -> list:
    if not snippets:
        return []
    seqs   = TOKENIZER.texts_to_sequences([clean_code(s) for s in snippets])
    padded = pad_sequences(seqs, maxlen=MAX_LEN,
                           padding="post", truncating="post")
    probs  = MODEL.predict(padded, verbose=0)
    out    = []
    for row in probs:
        top3 = np.argsort(row)[::-1][:3]
        lbl  = ENCODER.inverse_transform([top3[0]])[0]
        conf = float(row[top3[0]])
        meta = LABEL_META.get(lbl, ("UNKNOWN","N/A","N/A"))
        out.append({
            "label"     : lbl,
            "severity"  : meta[0],
            "cwe_id"    : meta[1],
            "owasp"     : meta[2],
            "confidence": round(conf * 100, 2),
            "uncertain" : conf < threshold,
            "top3"      : [
                (ENCODER.inverse_transform([i])[0],
                 round(float(row[i]) * 100, 2)) for i in top3
            ],
        })
    return out

print("✓ Predictor ready")


# ═════════════════════════════════════════════════════════════════
# CELL 8 — AST Call Graph  (Python only)
# ═════════════════════════════════════════════════════════════════

@dataclass
class FuncNode:
    name       : str
    file       : str
    lineno     : int
    end_lineno : int
    source     : str
    calls      : List[str] = field(default_factory=list)
    called_by  : List[str] = field(default_factory=list)


class _CGBuilder(ast.NodeVisitor):
    def __init__(self, lines, filepath):
        self.lines    = lines
        self.filepath = filepath
        self.funcs    : Dict[str, FuncNode] = {}
        self._cur     : Optional[str] = None

    def visit_FunctionDef(self, node):
        end = getattr(node, "end_lineno", node.lineno + 1)
        src = "\n".join(self.lines[node.lineno - 1 : end])
        self.funcs[node.name] = FuncNode(
            name       = node.name,
            file       = self.filepath,
            lineno     = node.lineno,
            end_lineno = end,
            source     = src,
        )
        prev, self._cur = self._cur, node.name
        self.generic_visit(node)
        self._cur = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        if self._cur:
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name:
                self.funcs[self._cur].calls.append(name)
        self.generic_visit(node)


def build_call_graph(source: str, filepath: str) -> Dict[str, FuncNode]:
    try:
        tree  = ast.parse(source)
        lines = source.splitlines()
        cg    = _CGBuilder(lines, filepath)
        cg.visit(tree)
        # Build reverse edges
        for fn in cg.funcs.values():
            for callee in fn.calls:
                if callee in cg.funcs:
                    cg.funcs[callee].called_by.append(fn.name)
        return cg.funcs
    except SyntaxError:
        return {}


def get_call_chain_context(func_name: str,
                            graph: Dict[str, FuncNode],
                            max_depth: int = 2) -> tuple:
    """
    Returns (combined_source, call_chain_list).
    combined_source = function source + sources of all callers up to max_depth.
    call_chain_list = ['caller2', 'caller1', 'func_name']
    """
    if func_name not in graph:
        return "", []

    visited    = set()
    parts      = []
    chain      = []

    def collect(name, depth):
        if depth < 0 or name in visited or name not in graph:
            return
        visited.add(name)
        fn = graph[name]
        parts.append(
            f"# ── function: {name}() ── line {fn.lineno} ──\n{fn.source}"
        )
        for caller in fn.called_by[:3]:
            chain.append(caller)
            collect(caller, depth - 1)

    collect(func_name, max_depth)
    return "\n\n".join(parts), list(dict.fromkeys(chain))  # deduplicated

print("✓ AST call graph builder ready")


# ═════════════════════════════════════════════════════════════════
# CELL 9 — File scanners
# ═════════════════════════════════════════════════════════════════

def _sliding_chunks(source: str, window: int = 6, step: int = 3):
    lines     = source.splitlines()
    non_blank = [(i, l) for i, l in enumerate(lines) if l.strip()]
    chunks    = []
    for i in range(0, max(1, len(non_blank) - window + 1), step):
        blk = non_blank[i:i + window]
        if blk:
            chunks.append(("\n".join(l for _, l in blk), blk[0][0] + 1))
    if len(non_blank) <= 40:
        chunks.insert(0, (source, 1))
    return chunks or [(source, 1)]


def scan_python(source: str, filepath: str, display_path: str,
                threshold: float, call_depth: int) -> list:
    findings = []
    graph    = build_call_graph(source, filepath)
    # Store for graph visualisation
    if graph:
        ALL_CALL_GRAPHS[filepath] = graph

    if graph:
        # Build contexts + run batch prediction
        func_names = list(graph.keys())
        contexts   = []
        chains     = []
        for name in func_names:
            ctx, chain = get_call_chain_context(name, graph, call_depth)
            contexts.append(ctx or graph[name].source)
            chains.append(chain)

        preds = predict_batch(contexts, threshold)

        for fname, ctx, chain, pred in zip(func_names, contexts, chains, preds):
            if pred["label"] == "SAFE" and not SCAN_CONFIG["show_safe"]:
                continue
            fn = graph[fname]
            findings.append({
                "file"         : display_path,
                "language"     : "python",
                "function_name": fname,
                "line_start"   : fn.lineno,
                "line_end"     : fn.end_lineno,
                "call_chain"   : chain,
                "snippet"      : ctx[:500],
                "analysis_mode": "ast_call_graph",
                **pred,
            })
    else:
        # Fallback: sliding window
        chunks = _sliding_chunks(source)
        texts  = [c[0] for c in chunks]
        preds  = predict_batch(texts, threshold)
        lines  = source.splitlines()
        for (chunk_text, start), pred in zip(chunks, preds):
            if pred["label"] == "SAFE" and not SCAN_CONFIG["show_safe"]:
                continue
            findings.append({
                "file"         : display_path,
                "language"     : "python",
                "function_name": None,
                "line_start"   : start,
                "line_end"     : min(start + chunk_text.count("\n"), len(lines)),
                "call_chain"   : [],
                "snippet"      : chunk_text[:500],
                "analysis_mode": "sliding_window",
                **pred,
            })
    return findings


def scan_generic(source: str, display_path: str,
                 lang: str, threshold: float) -> list:
    findings = []
    chunks   = _sliding_chunks(source)
    texts    = [c[0] for c in chunks]
    preds    = predict_batch(texts, threshold)
    lines    = source.splitlines()
    for (chunk_text, start), pred in zip(chunks, preds):
        if pred["label"] == "SAFE" and not SCAN_CONFIG["show_safe"]:
            continue
        findings.append({
            "file"         : display_path,
            "language"     : lang,
            "function_name": None,
            "line_start"   : start,
            "line_end"     : min(start + chunk_text.count("\n"), len(lines)),
            "call_chain"   : [],
            "snippet"      : chunk_text[:500],
            "analysis_mode": "sliding_window",
            **pred,
        })
    return findings


def scan_folder(target: Path, threshold: float, call_depth: int):
    pairs = []
    for root, dirs, fnames in os.walk(target):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in fnames:
            fp  = Path(root) / fname
            ext = fp.suffix.lower()
            if ext in LANG_MAP:
                try:    rel = str(fp.relative_to(target))
                except  ValueError: rel = str(fp)
                pairs.append((fp, rel))
    pairs.sort(key=lambda x: x[1])

    total    = len(pairs)
    findings = []

    for idx, (fp, display) in enumerate(pairs, 1):
        lang = LANG_MAP[fp.suffix.lower()]
        print(f"  [{idx:>4}/{total}]  {display:<65}", end="\r", flush=True)
        try:
            source = fp.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"\n  [SKIP] {display} — {e}")
            continue

        if lang == "python":
            findings += scan_python(source, str(fp), display,
                                    threshold, call_depth)
        else:
            findings += scan_generic(source, display, lang, threshold)

    print(f"\n  Done: {total} file(s) scanned, {len(findings)} finding(s)")
    return findings, total

print("✓ Scanners ready")


# ═════════════════════════════════════════════════════════════════
# CELL 10 — Report builder
# ═════════════════════════════════════════════════════════════════
W = 72

def build_report(findings: list, target_name: str,
                 scanned_files: int, elapsed: float) -> str:

    sev_c  = Counter(f["severity"]                      for f in findings)
    lbl_c  = Counter(f["label"]                         for f in findings)
    file_c = Counter(f["file"]                          for f in findings)
    mode_c = Counter(f.get("analysis_mode", "?")        for f in findings)
    total  = len(findings)

    risk = min(100,
               sev_c.get("CRITICAL",0) * 10 +
               sev_c.get("HIGH",0)     *  5 +
               sev_c.get("MEDIUM",0)   *  2)
    rl   = ("CRITICAL RISK" if risk >= 70 else
            "HIGH RISK"     if risk >= 40 else
            "MEDIUM RISK"   if risk >  0  else "LOW / CLEAN")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L   = []; a = L.append

    def hr(c="═"): return c * W
    def sec(t):
        a(f"\n{hr()}")
        a(f"  {t}")
        a(hr())

    # ── Header ─────────────────────────────────────────────
    a(hr())
    a("  AI-SAST SMART SECURITY REPORT  v2.1")
    a("  Mitigata Security  |  VulnDetector (CNN + Stacked BiLSTM + MHA)")
    a("  github.com/gijutsu-hub/aisastscanner")
    a(hr())
    a(f"  Target          : {target_name}")
    a(f"  Generated       : {now}")
    a(f"  Scan time       : {elapsed:.1f}s")
    a(f"  Files scanned   : {scanned_files}")
    a(f"  Confidence thr. : {SCAN_CONFIG['threshold']}")
    a(f"  Call depth      : {SCAN_CONFIG['call_depth']}")
    a(hr("─"))
    a(f"  Analysis modes used:")
    for mode, cnt in sorted(mode_c.items(), key=lambda x: -x[1]):
        a(f"    {mode:<28} : {cnt} finding(s)")

    # ── Risk score ─────────────────────────────────────────
    sec("RISK SCORE")
    a(f"  Score       : {risk} / 100")
    a(f"  Risk level  : {rl}")
    a(f"  Formula     : CRITICAL×10 + HIGH×5 + MEDIUM×2")
    a("")
    a(f"  CRITICAL    : {sev_c.get('CRITICAL', 0):>4}  findings")
    a(f"  HIGH        : {sev_c.get('HIGH',     0):>4}  findings")
    a(f"  MEDIUM      : {sev_c.get('MEDIUM',   0):>4}  findings")
    a(f"  TOTAL       : {total:>4}  findings across {len(file_c)} file(s)")

    # ── OWASP ──────────────────────────────────────────────
    sec("OWASP TOP 10 BREAKDOWN")
    owasp_c = Counter(f["owasp"] for f in findings if f["owasp"] != "N/A")
    if owasp_c:
        mx = max(owasp_c.values())
        for cat, cnt in sorted(owasp_c.items(), key=lambda x: -x[1]):
            bar = "█" * int(cnt / max(mx, 1) * 22)
            a(f"  {cnt:>3}  {bar:<22}  {cat}")
    else:
        a("  No OWASP-mapped findings.")

    # ── Vulnerability summary ───────────────────────────────
    sec("VULNERABILITY TYPE SUMMARY")
    a(f"  {'Type':<30} {'Count':>5}  {'Severity':<10}  CWE")
    a(f"  {'─'*30} {'─'*5}  {'─'*10}  {'─'*10}")
    for lbl, cnt in lbl_c.most_common():
        m = LABEL_META.get(lbl, ("UNKNOWN","N/A","N/A"))
        a(f"  {lbl:<30} {cnt:>5}  {m[0]:<10}  {m[1]}")

    # ── Most affected files ────────────────────────────────
    sec("MOST AFFECTED FILES")
    for fp, cnt in file_c.most_common(10):
        a(f"  {cnt:>3} finding(s)   {fp}")

    # ── Cross-function findings summary ─────────────────────
    cross_fn = [f for f in findings if f.get("call_chain")]
    if cross_fn:
        sec("CROSS-FUNCTION TAINT PATHS DETECTED")
        a(f"  {len(cross_fn)} finding(s) involve tainted data flowing")
        a(f"  across function call boundaries.\n")
        for f in cross_fn[:10]:
            chain = f.get("call_chain", [])
            fn    = f.get("function_name", "?")
            path  = " → ".join(chain + [fn]) if chain else fn
            a(f"  [{f['severity']:<8}]  {f['label']:<25}  {f['file']}")
            a(f"             Taint path: {path}")
            a("")

    # ── Detailed findings ──────────────────────────────────
    sec("DETAILED FINDINGS")
    sorted_f = sorted(
        findings,
        key=lambda f: (SEV_ORDER.get(f["severity"], 4), -f["confidence"])
    )

    for i, f in enumerate(sorted_f, 1):
        unc  = "  ⚠ LOW CONFIDENCE" if f["uncertain"] else ""
        mode = f.get("analysis_mode", "?")
        fn   = f.get("function_name")
        chain= f.get("call_chain", [])

        a("")
        a(f"  ┌─ FINDING #{i} {'─' * (W - 14)}")
        a(f"  │  Severity       : {f['severity']}{unc}")
        a(f"  │  Type           : {f['label']}")
        a(f"  │  CWE            : {f['cwe_id']}")
        a(f"  │  OWASP          : {f['owasp']}")
        a(f"  │  Confidence     : {f['confidence']:.1f}%")
        a(f"  │  File           : {f['file']}")
        a(f"  │  Lines          : {f['line_start']} – {f['line_end']}")
        a(f"  │  Language       : {f['language']}")
        a(f"  │  Analysis       : {mode}")

        if fn:
            a(f"  │  Function       : {fn}()")

        if chain:
            taint_path = " → ".join(chain + [fn]) if fn else " → ".join(chain)
            a(f"  │  Taint path     : {taint_path}")
            a(f"  │  ⚑ Cross-function taint detected — data flows from")
            a(f"  │    caller into this function without sanitization")

        a(f"  │")
        a(f"  │  Top-3 Predictions:")
        for rank, (lbl, pct) in enumerate(f["top3"], 1):
            bar = "█" * int(pct / 5)
            a(f"  │    {rank}. {lbl:<30} {pct:6.2f}%  {bar}")

        a(f"  │")
        a(f"  │  Code Context  [{mode}]:")
        snippet_lines = f["snippet"].splitlines()
        for sl in snippet_lines[:25]:
            a(f"  │    {sl}")
        if len(snippet_lines) > 25:
            a(f"  │    ... ({len(snippet_lines)-25} more lines)")
        a(f"  └{'─' * (W - 3)}")

    # ── Footer ─────────────────────────────────────────────
    a("")
    a(hr())
    a("  DISCLAIMER")
    a("  This is AI-assisted triage output. All findings must be")
    a("  validated by a qualified security engineer before action.")
    a(hr("─"))
    a(f"  Engine  : VulnDetector  |  MAX_LEN={MAX_LEN}  |  Classes={len(ENCODER.classes_)}")
    a(f"  Source  : github.com/gijutsu-hub/aisastscanner")
    a(hr())

    return "\n".join(L)

print("✓ Report builder ready")


# ═════════════════════════════════════════════════════════════════
# CELL 10b — Call Graph Visualiser
#   Builds an interactive-style node/edge graph of every function
#   in the codebase, coloured by vulnerability grade.
#   Saves PNG + shows inline in Colab.
# ═════════════════════════════════════════════════════════════════

# ── Severity palette ────────────────────────────────────────────
NODE_COLORS = {
    "CRITICAL" : "#e74c3c",   # red
    "HIGH"     : "#e67e22",   # orange
    "MEDIUM"   : "#f1c40f",   # yellow
    "SAFE"     : "#2ecc71",   # green
    "UNKNOWN"  : "#95a5a6",   # grey   (not scanned / no result)
}

GRADE_MAP = {
    "CRITICAL" : "F",
    "HIGH"     : "D",
    "MEDIUM"   : "C",
    "SAFE"     : "A",
    "UNKNOWN"  : "?",
}


def build_viz_graph(all_graphs, findings):
    """
    all_graphs : { filepath: {func_name: FuncNode} }
    findings   : list of finding dicts from scan_folder()

    Returns a networkx DiGraph with node attributes:
        severity, label, grade, file, color, size
    """
    G = nx.DiGraph()

    # Index findings by (file, function_name)
    finding_idx = {}
    for f in findings:
        key = (f["file"], f.get("function_name"))
        # Keep the worst severity per node
        if key not in finding_idx:
            finding_idx[key] = f
        else:
            existing = finding_idx[key]
            if SEV_ORDER.get(f["severity"],4) < SEV_ORDER.get(existing["severity"],4):
                finding_idx[key] = f

    # Add nodes
    for filepath, graph in all_graphs.items():
        for fname, fn in graph.items():
            display = str(Path(filepath).relative_to(
                Path(SCAN_CONFIG["target"])) if Path(filepath).is_absolute()
                else Path(filepath))
            key = (display, fname)
            f   = finding_idx.get(key)
            sev  = f["severity"]  if f else "UNKNOWN"
            vuln = f["label"]     if f else "SAFE"
            conf = f["confidence"]if f else 0.0

            G.add_node(
                fname,
                severity   = sev,
                vuln_label = vuln,
                grade      = GRADE_MAP.get(sev, "?"),
                confidence = conf,
                file       = display,
                color      = NODE_COLORS.get(sev, NODE_COLORS["UNKNOWN"]),
                # Node size scales with confidence (bigger = more certain)
                size       = 800 + int(conf * 20),
            )

    # Add edges (call relationships)
    for filepath, graph in all_graphs.items():
        for fname, fn in graph.items():
            if fname not in G:
                continue
            for callee in fn.calls:
                if callee in G:
                    G.add_edge(fname, callee)

    return G


def draw_call_graph(G, title="Function Call Graph — Vulnerability Map",
                    save_path="call_graph.png"):
    if len(G.nodes) == 0:
        print("  [INFO] No functions to visualise (no Python AST graphs built).")
        return

    fig_w  = max(18, len(G.nodes) * 0.6)
    fig_h  = max(12, len(G.nodes) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h),
                           facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.set_title(title, color="white", fontsize=15, fontweight="bold", pad=18)

    # ── Layout ────────────────────────────────────────────
    # Try hierarchical (dot) if graphviz available, else spring
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=2.5)

    # ── Draw edges ────────────────────────────────────────
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color  = "#444c56",
        arrows      = True,
        arrowsize   = 18,
        arrowstyle  = "-|>",
        width       = 1.2,
        alpha       = 0.7,
        connectionstyle = "arc3,rad=0.08",
    )

    # ── Draw nodes grouped by severity ───────────────────
    for sev, color in NODE_COLORS.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get("severity") == sev]
        if not nodes:
            continue
        sizes  = [G.nodes[n].get("size", 800) for n in nodes]
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist   = nodes,
            node_color = color,
            node_size  = sizes,
            alpha      = 0.92,
            linewidths = 2,
            edgecolors = "#ffffff33",
        )

    # ── Node labels  (name + grade) ────────────────────────
    labels = {}
    for n, d in G.nodes(data=True):
        grade = d.get("grade", "?")
        vuln  = d.get("vuln_label", "")
        short = n[:18] + "…" if len(n) > 18 else n
        labels[n] = f"{short}\n[{grade}] {vuln[:12] if vuln != 'SAFE' else '✓'}"

    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size  = 7,
        font_color = "white",
        font_weight= "bold",
        bbox       = dict(boxstyle="round,pad=0.2",
                          fc="#00000066", ec="none"),
    )

    # ── Legend ────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=NODE_COLORS["CRITICAL"], label="CRITICAL (Grade F)"),
        mpatches.Patch(color=NODE_COLORS["HIGH"],     label="HIGH     (Grade D)"),
        mpatches.Patch(color=NODE_COLORS["MEDIUM"],   label="MEDIUM   (Grade C)"),
        mpatches.Patch(color=NODE_COLORS["SAFE"],     label="SAFE     (Grade A)"),
        mpatches.Patch(color=NODE_COLORS["UNKNOWN"],  label="NOT SCANNED"),
        Line2D([0],[0], color="#444c56", linewidth=1.5,
               label="Calls →"),
    ]
    ax.legend(handles=legend_items, loc="upper left",
              facecolor="#161b22", edgecolor="#444c56",
              labelcolor="white", fontsize=9, framealpha=0.9)

    # ── Grade summary text box ────────────────────────────
    grade_counts = Counter(d.get("grade","?") for _, d in G.nodes(data=True))
    summary = "  ".join(f"{g}:{c}" for g,c in sorted(grade_counts.items()))
    ax.text(0.99, 0.01, f"Grade summary: {summary}",
            transform=ax.transAxes, ha="right", va="bottom",
            color="#8b949e", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#161b22", ec="#444c56"))

    ax.axis("off")
    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.show()
    print(f"  ✓ Graph saved → {save_path}")


def draw_per_file_graphs(all_graphs, findings, max_files=8):
    """Draw one smaller call graph per file (top-N most affected)."""
    finding_idx = {}
    for f in findings:
        key = (f["file"], f.get("function_name"))
        if key not in finding_idx or SEV_ORDER.get(f["severity"],4) <                 SEV_ORDER.get(finding_idx[key]["severity"],4):
            finding_idx[key] = f

    # Rank files by worst severity
    file_sev = defaultdict(lambda: 10)
    for f in findings:
        s = SEV_ORDER.get(f["severity"], 4)
        if s < file_sev[f["file"]]:
            file_sev[f["file"]] = s

    sorted_files = sorted(file_sev, key=lambda fp: file_sev[fp])[:max_files]

    for filepath, graph in all_graphs.items():
        if not graph:
            continue
        try:
            display = str(Path(filepath).relative_to(
                Path(SCAN_CONFIG["target"])))
        except ValueError:
            display = filepath

        if display not in sorted_files:
            continue

        G = nx.DiGraph()
        for fname, fn in graph.items():
            key = (display, fname)
            f   = finding_idx.get(key)
            sev  = f["severity"]  if f else "UNKNOWN"
            vuln = f["label"]     if f else "SAFE"
            conf = f["confidence"]if f else 0.0
            G.add_node(fname,
                severity=sev, vuln_label=vuln, grade=GRADE_MAP.get(sev,"?"),
                confidence=conf, file=display,
                color=NODE_COLORS.get(sev, NODE_COLORS["UNKNOWN"]),
                size=1200 + int(conf * 25))
        for fname, fn in graph.items():
            for callee in fn.calls:
                if callee in G and fname in G:
                    G.add_edge(fname, callee)

        if len(G.nodes) < 2:
            continue

        short_name = Path(display).name
        draw_call_graph(
            G,
            title    = f"Call Graph: {short_name}",
            save_path= f"graph_{re.sub(r'[^a-zA-Z0-9]', '_', short_name)}.png",
        )


# Storage for call graphs across all files (populated during scan)
ALL_CALL_GRAPHS: Dict[str, Dict] = {}

print("✓ Graph visualiser ready")


# ═════════════════════════════════════════════════════════════════
# CELL 11 — ▶▶  RUN THE SCAN  ◀◀
# ═════════════════════════════════════════════════════════════════
target_name = zip_name.replace(".zip", "")

print("\n" + "═"*72)
print("  AI-SAST Smart Scanner v2.1  |  Mitigata Security")
print("═"*72)
print(f"  Target     : {target_name}")
print(f"  Model      : {MODEL_PATH.name}")
print(f"  MAX_LEN    : {MAX_LEN}")
print(f"  Threshold  : {SCAN_CONFIG['threshold']}")
print(f"  Call depth : {SCAN_CONFIG['call_depth']}")
print("═"*72 + "\n")

t0 = datetime.now()
findings, scanned_files = scan_folder(
    Path(SCAN_CONFIG["target"]),
    threshold  = SCAN_CONFIG["threshold"],
    call_depth = SCAN_CONFIG["call_depth"],
)
elapsed = (datetime.now() - t0).total_seconds()

# ── Console summary ────────────────────────────────────────────
sev_c    = Counter(f["severity"] for f in findings)
cross_fn = sum(1 for f in findings if f.get("call_chain"))

print(f"\n{'═'*72}")
print(f"  Scan complete    : {elapsed:.1f}s")
print(f"  Files scanned    : {scanned_files}")
print(f"  Total findings   : {len(findings)}")
print(f"    CRITICAL       : {sev_c.get('CRITICAL', 0)}")
print(f"    HIGH           : {sev_c.get('HIGH',     0)}")
print(f"    MEDIUM         : {sev_c.get('MEDIUM',   0)}")
print(f"  Cross-func taints: {cross_fn}")
print("═"*72)

# ── 1. Call Graph Visualisation ───────────────────────────────
print("\n  Building call graph visualisation ...")

if ALL_CALL_GRAPHS:
    # Full codebase graph
    G_full = build_viz_graph(ALL_CALL_GRAPHS, findings)
    print(f"  Nodes: {len(G_full.nodes)}  |  Edges: {len(G_full.edges)}")

    grade_counts = Counter(
        GRADE_MAP.get(d.get("severity","UNKNOWN"), "?")
        for _, d in G_full.nodes(data=True)
    )
    print("  Grade distribution:")
    for grade in ["F","D","C","A","?"]:
        cnt = grade_counts.get(grade, 0)
        bar = "█" * cnt
        print(f"    {grade} : {cnt:>4}  {bar}")

    # Full graph
    draw_call_graph(
        G_full,
        title     = f"Full Codebase Call Graph — {target_name}",
        save_path = "call_graph_full.png",
    )

    # Per-file graphs (top 8 most vulnerable files)
    print("\n  Drawing per-file call graphs ...")
    draw_per_file_graphs(ALL_CALL_GRAPHS, findings, max_files=8)

    # Download graphs
    if SCAN_CONFIG["auto_download"]:
        from google.colab import files as cf
        import glob
        for png in glob.glob("*.png"):
            cf.download(png)
            print(f"  ✓ Downloaded → {png}")
else:
    print("  [INFO] No Python AST call graphs built (no Python files or all parse errors).")

# ── 2. Build & print text report ──────────────────────────────
report = build_report(findings, target_name, scanned_files, elapsed)
print("\n\n" + report)

# ── Save report ────────────────────────────────────────────────
report_file = SCAN_CONFIG["report_file"]
with open(report_file, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\n✓ Report saved → {report_file}")

# ── Auto-download ──────────────────────────────────────────────
if SCAN_CONFIG["auto_download"]:
    from google.colab import files as cf
    cf.download(report_file)
    print(f"✓ Downloaded → {report_file}")

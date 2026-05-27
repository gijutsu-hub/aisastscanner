"""
AI-Powered SAST Vulnerability Detector  v3.0
Dataset : sast_training_expanded.json  (25,124 entries, 35 classes)
Author  : Mitigata Security – VAPT Team
"""

# !pip install -q tensorflow scikit-learn pandas matplotlib seaborn

# ── Silence TF/CUDA double-registration noise ────────────────────
# Must be set BEFORE any tensorflow import.
#   TF_CPP_MIN_LOG_LEVEL  : 0=all  1=no INFO  2=no WARNING  3=no ERROR
#   CUDA_MODULE_LOADING   : LAZY  → skips redundant plugin scans
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]       = "3"   # suppress cuFFT/cuDNN/cuBLAS msgs
os.environ["CUDA_MODULE_LOADING"]         = "LAZY"
os.environ["TF_ENABLE_ONEDNN_OPTS"]       = "0"   # suppress oneDNN info line
os.environ["TF_XLA_FLAGS"]               = "--tf_xla_enable_xla_devices=false"

# absl logging must be configured before TF pulls it in
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import warnings
warnings.filterwarnings("ignore")
# ─────────────────────────────────────────────────────────────────

import json, re, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

print("TensorFlow :", tf.__version__)
print("GPU        :", tf.config.list_physical_devices("GPU"))


# ============================================================
# 1.  CONFIG
#     Tuned for the expanded 25k / 35-class dataset.
#     Key changes vs v2:
#       • json_file     → sast_training_expanded.json
#       • vocab_size    → 5 000  (actual unique tokens ≈ 1 464)
#       • max_len       → 32     (p99 of token lengths = 18; 32 gives headroom)
#       • embed_dim     → 128    (unchanged)
#       • dense_units   → [512, 256, 128]  (wider head for 35 classes)
#       • dropout_rate  → 0.45
#       • epochs        → 60     (more classes → more epochs budget)
#       • label_smooth  → 0.05   (lighter smoothing; classes are well-separated)
# ============================================================

CFG = {
    # ── Data ──────────────────────────────────────────────
    "json_file"    : "./sast_training_expanded.json",
    "test_size"    : 0.15,
    "val_size"     : 0.15,

    # ── Tokeniser ─────────────────────────────────────────
    "vocab_size"   : 5_000,    # ← actual unique tokens ≈ 1 464; ceiling kept for OOV
    "max_len"      : 32,       # ← p99 of dataset token lengths = 18; 32 gives headroom

    # ── Embedding ─────────────────────────────────────────
    "embed_dim"    : 128,
    "embed_dropout": 0.25,

    # ── CNN branch ────────────────────────────────────────
    "cnn_filters"  : 128,
    "cnn_kernels"  : [2, 3, 5],

    # ── BiLSTM branch ─────────────────────────────────────
    "lstm_units"   : 128,

    # ── Multi-Head Attention ───────────────────────────────
    "attn_heads"   : 4,
    "attn_key_dim" : 32,

    # ── Dense head  (wider for 35 classes) ────────────────
    "dense_units"  : [512, 256, 128],
    "dropout_rate" : 0.45,
    "l2_reg"       : 1e-4,

    # ── Training ──────────────────────────────────────────
    "batch_size"   : 128,      # ← larger batch; dataset is 2.5× bigger
    "epochs"       : 60,
    "lr"           : 1e-3,
    "label_smooth" : 0.05,     # ← lighter smoothing; classes are well-separated

    # ── Artefacts ─────────────────────────────────────────
    "model_path"   : "vuln_model_v3.keras",
    "tok_path"     : "tokenizer_v3.pkl",
    "enc_path"     : "label_encoder_v3.pkl",
}

# ── Full severity / CWE / OWASP maps (all 35 labels) ─────
LABEL_META = {
    # label                    : (severity,  cwe_id,    owasp)
    "SQLI"                     : ("CRITICAL", "CWE-89",  "A03:2021 – Injection"),
    "XSS"                      : ("HIGH",     "CWE-79",  "A03:2021 – Injection"),
    "COMMAND_INJECTION"        : ("CRITICAL", "CWE-78",  "A03:2021 – Injection"),
    "RCE"                      : ("CRITICAL", "CWE-94",  "A03:2021 – Injection"),
    "BUFFER_OVERFLOW"          : ("HIGH",     "CWE-120", "A06:2021 – Vulnerable Components"),
    "FORMAT_STRING"            : ("HIGH",     "CWE-134", "A03:2021 – Injection"),
    "PATH_TRAVERSAL"           : ("HIGH",     "CWE-22",  "A01:2021 – Broken Access Control"),
    "XXE"                      : ("HIGH",     "CWE-611", "A05:2021 – Security Misconfiguration"),
    "DESERIALIZATION"          : ("CRITICAL", "CWE-502", "A08:2021 – Insecure Deserialization"),
    "HARDCODED_SECRET"         : ("CRITICAL", "CWE-798", "A07:2021 – Identification & Auth Failures"),
    "WEAK_CRYPTO"              : ("MEDIUM",   "CWE-326", "A02:2021 – Cryptographic Failures"),
    "SSL_BYPASS"               : ("HIGH",     "CWE-295", "A02:2021 – Cryptographic Failures"),
    "OPEN_REDIRECT"            : ("MEDIUM",   "CWE-601", "A01:2021 – Broken Access Control"),
    "SSRF"                     : ("HIGH",     "CWE-918", "A10:2021 – SSRF"),
    "LDAP_INJECTION"           : ("HIGH",     "CWE-90",  "A03:2021 – Injection"),
    "XPATH_INJECTION"          : ("HIGH",     "CWE-643", "A03:2021 – Injection"),
    "SSTI"                     : ("CRITICAL", "CWE-1336","A03:2021 – Injection"),
    "LOG_INJECTION"            : ("MEDIUM",   "CWE-117", "A09:2021 – Security Logging Failures"),
    "IDOR"                     : ("HIGH",     "CWE-639", "A01:2021 – Broken Access Control"),
    "CSRF"                     : ("MEDIUM",   "CWE-352", "A01:2021 – Broken Access Control"),
    "INSECURE_FILE_UPLOAD"     : ("HIGH",     "CWE-434", "A04:2021 – Insecure Design"),
    "INTEGER_OVERFLOW"         : ("HIGH",     "CWE-190", "A06:2021 – Vulnerable Components"),
    "USE_AFTER_FREE"           : ("HIGH",     "CWE-416", "A06:2021 – Vulnerable Components"),
    "NULL_DEREFERENCE"         : ("MEDIUM",   "CWE-476", "A06:2021 – Vulnerable Components"),
    "RACE_CONDITION"           : ("HIGH",     "CWE-362", "A04:2021 – Insecure Design"),
    "MASS_ASSIGNMENT"          : ("HIGH",     "CWE-915", "A04:2021 – Insecure Design"),
    "JWT_WEAKNESS"             : ("CRITICAL", "CWE-347", "A07:2021 – Identification & Auth Failures"),
    "CORS_MISCONFIGURATION"    : ("MEDIUM",   "CWE-942", "A05:2021 – Security Misconfiguration"),
    "REGEX_DOS"                : ("MEDIUM",   "CWE-1333","A04:2021 – Insecure Design"),
    "HTTP_RESPONSE_SPLITTING"  : ("MEDIUM",   "CWE-113", "A03:2021 – Injection"),
    "SESSION_FIXATION"         : ("HIGH",     "CWE-384", "A07:2021 – Identification & Auth Failures"),
    "PRIVILEGE_ESCALATION"     : ("CRITICAL", "CWE-269", "A01:2021 – Broken Access Control"),
    "FILE_INCLUSION"           : ("CRITICAL", "CWE-98",  "A03:2021 – Injection"),
    "SENSITIVE_DATA_EXPOSURE"  : ("HIGH",     "CWE-312", "A02:2021 – Cryptographic Failures"),
    "SAFE"                     : ("NONE",     "N/A",     "N/A"),
}

SEVERITY = {lbl: meta[0] for lbl, meta in LABEL_META.items()}
CWE_MAP  = {lbl: meta[1] for lbl, meta in LABEL_META.items()}
OWASP    = {lbl: meta[2] for lbl, meta in LABEL_META.items()}


# ============================================================
# 2.  LOAD & DEDUPLICATE
#     The expanded dataset has extra fields (language, severity,
#     cwe_id, owasp_category, multiline, source) — we only need
#     "code" and "label" for training.
# ============================================================

with open(CFG["json_file"], "r") as f:
    raw = json.load(f)

df = pd.DataFrame(raw).dropna(subset=["code", "label"]) \
       .drop_duplicates(subset=["code"])

print(f"\nLoaded  : {len(df):,} entries")
print(f"Labels  : {df['label'].nunique()}")
print(df["label"].value_counts().to_string())

# Optional: show language breakdown if column exists
if "language" in df.columns:
    print(f"\nLanguages:\n{df['language'].value_counts().to_string()}")

if "source" in df.columns:
    print(f"\nSources:\n{df['source'].value_counts().to_string()}")


# ============================================================
# 3.  PRE-PROCESSING
#     Identical tokenisation strategy as v2, but we also handle
#     multi-line snippets (strip newlines → space) before the
#     symbol-splitting pass.
# ============================================================

def clean_code(snippet: str) -> str:
    # Normalise multi-line snippets to a single line first
    snippet = snippet.replace("\n", " ").replace("\r", " ")
    snippet = snippet.lower().strip()
    # Space-pad every operator / delimiter so they become separate tokens
    snippet = re.sub(r"([(){}\[\];:,.<>+\-*/=!&|^~%@#$])", r" \1 ", snippet)
    snippet = re.sub(r"\s+", " ", snippet)
    return snippet


df["code_clean"] = df["code"].apply(clean_code)
codes  = df["code_clean"].values
labels = df["label"].values


# ============================================================
# 4.  ENCODE LABELS
# ============================================================

encoder   = LabelEncoder()
y_int     = encoder.fit_transform(labels)
n_classes = len(encoder.classes_)
y_onehot  = tf.keras.utils.to_categorical(y_int, num_classes=n_classes)

print(f"\nClasses ({n_classes}): {list(encoder.classes_)}")


# ============================================================
# 5.  TOKENISE  (vocab_size tuned to actual corpus)
# ============================================================

tokenizer = Tokenizer(
    num_words = CFG["vocab_size"],
    oov_token = "<OOV>",
    filters   = "",   # keep all punctuation — already spaced above
    lower     = False,
)
tokenizer.fit_on_texts(codes)

X = tokenizer.texts_to_sequences(codes)
X = pad_sequences(X, maxlen=CFG["max_len"], padding="post", truncating="post")

print(f"\nVocab   : {min(len(tokenizer.word_index)+1, CFG['vocab_size']):,}")
print(f"X shape : {X.shape}")


# ============================================================
# 6.  STRATIFIED SPLIT  (train / val / test)
# ============================================================

X_tmp, X_test, y_tmp_oh, y_test_oh, y_tmp_int, y_test_int = train_test_split(
    X, y_onehot, y_int,
    test_size    = CFG["test_size"],
    random_state = 42,
    stratify     = y_int,
)
X_train, X_val, y_train, y_val, y_train_int, y_val_int = train_test_split(
    X_tmp, y_tmp_oh, y_tmp_int,
    test_size    = CFG["val_size"] / (1 - CFG["test_size"]),
    random_state = 42,
    stratify     = y_tmp_int,
)
print(f"\nTrain : {len(X_train):,}  |  Val : {len(X_val):,}  |  Test : {len(X_test):,}")


# ============================================================
# 7.  CLASS WEIGHTS  (handles any remaining imbalance)
# ============================================================

cw_values     = compute_class_weight(
    class_weight = "balanced",
    classes      = np.unique(y_train_int),
    y            = y_train_int,
)
class_weights = dict(enumerate(cw_values))


# ============================================================
# 8.  MODEL  — CNN + BiLSTM + Multi-Head Attention
#     Bug fix vs v2: removed stray `x` token after conv block.
#     Architecture unchanged; dense head widened for 35 classes.
# ============================================================

def build_model(vocab_size, embed_dim, max_len, n_classes, cfg):

    inp = Input(shape=(max_len,), name="input")

    # ── Embedding ─────────────────────────────────────────
    emb = layers.Embedding(vocab_size, embed_dim, name="emb")(inp)
    emb = layers.SpatialDropout1D(cfg["embed_dropout"], name="emb_drop")(emb)

    # ── CNN branch (multi-scale) ───────────────────────────
    pools = []
    for k in cfg["cnn_kernels"]:
        x = layers.Conv1D(
            cfg["cnn_filters"], k, padding="same",
            activation="relu", kernel_regularizer=l2(cfg["l2_reg"]),
            name=f"conv_{k}"
        )(emb)                                    # ← fixed: removed stray `x`
        x = layers.GlobalMaxPooling1D(name=f"pool_{k}")(x)
        pools.append(x)

    cnn = layers.Concatenate(name="cnn_cat")(pools)
    cnn = layers.Dense(128, activation="relu", name="cnn_fc")(cnn)
    cnn = layers.BatchNormalization(name="cnn_bn")(cnn)
    cnn = layers.Dropout(cfg["dropout_rate"], name="cnn_drop")(cnn)

    # ── BiLSTM + Multi-Head Attention branch ──────────────
    lstm = layers.Bidirectional(
        layers.LSTM(cfg["lstm_units"], return_sequences=True, dropout=0.2),
        name="bilstm"
    )(emb)

    attn, _ = layers.MultiHeadAttention(
        num_heads=cfg["attn_heads"], key_dim=cfg["attn_key_dim"], name="mha"
    )(lstm, lstm, return_attention_scores=True)

    attn = layers.Add(name="residual")([lstm, attn])
    attn = layers.LayerNormalization(name="ln")(attn)
    seq  = layers.GlobalAveragePooling1D(name="gap")(attn)
    seq  = layers.Dense(128, activation="relu", name="seq_fc")(seq)
    seq  = layers.BatchNormalization(name="seq_bn")(seq)
    seq  = layers.Dropout(cfg["dropout_rate"], name="seq_drop")(seq)

    # ── Merge + Dense Head (widened for 35 classes) ───────
    x = layers.Concatenate(name="merge")([cnn, seq])

    for i, units in enumerate(cfg["dense_units"]):
        x = layers.Dense(
            units, activation="relu",
            kernel_regularizer=l2(cfg["l2_reg"]),
            name=f"fc_{i}"
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(cfg["dropout_rate"], name=f"drop_{i}")(x)

    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    return Model(inp, out, name="VulnDetector_v3")


model = build_model(
    vocab_size = CFG["vocab_size"],
    embed_dim  = CFG["embed_dim"],
    max_len    = CFG["max_len"],
    n_classes  = n_classes,
    cfg        = CFG,
)
model.summary()


# ============================================================
# 9.  COMPILE
# ============================================================

model.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate = CFG["lr"],
        clipnorm      = 1.0,
    ),
    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing = CFG["label_smooth"],
    ),
    metrics = ["accuracy"],
)


# ============================================================
# 10.  CALLBACKS
# ============================================================

callbacks = [
    EarlyStopping(
        monitor              = "val_accuracy",
        patience             = 10,           # ← more patience for 35 classes
        restore_best_weights = True,
        verbose              = 1,
    ),
    ModelCheckpoint(
        filepath       = CFG["model_path"],
        monitor        = "val_accuracy",
        save_best_only = True,
        verbose        = 1,
    ),
    ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.5,
        patience = 4,
        min_lr   = 1e-6,
        verbose  = 1,
    ),
]


# ============================================================
# 11.  TRAIN
# ============================================================

history = model.fit(
    X_train, y_train,
    validation_data = (X_val, y_val),
    epochs          = CFG["epochs"],
    batch_size      = CFG["batch_size"],
    class_weight    = class_weights,
    callbacks       = callbacks,
    verbose         = 1,
)


# ============================================================
# 12.  EVALUATE
# ============================================================

test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"\n{'='*52}")
print(f"  Test Accuracy : {test_acc * 100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")
print(f"{'='*52}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test_oh, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_true, y_pred,
    target_names = encoder.classes_,
    digits       = 4,
))


# ============================================================
# 13.  PLOTS
# ============================================================

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (train_key, val_key), title in zip(
    axes,
    [("accuracy", "val_accuracy"), ("loss", "val_loss")],
    ["Accuracy", "Loss"],
):
    ax.plot(history.history[train_key], label="Train", linewidth=2)
    ax.plot(history.history[val_key],   label="Val",   linewidth=2)
    ax.set_title(f"{title} — VulnDetector v3", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves_v3.png", dpi=150)
plt.show()

# Confusion matrix  (35×35 — use smaller font)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(22, 18))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels = encoder.classes_,
    yticklabels = encoder.classes_,
    annot_kws   = {"size": 7},
)
plt.title("Confusion Matrix — Test Set  (VulnDetector v3)", fontsize=13)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0,  fontsize=8)
plt.tight_layout()
plt.savefig("confusion_matrix_v3.png", dpi=150)
plt.show()

# Per-label F1 bar chart  (useful for 35 classes)
from sklearn.metrics import f1_score
f1_scores = f1_score(y_true, y_pred, average=None)
f1_df = pd.DataFrame({
    "label": encoder.classes_,
    "f1"   : f1_scores,
}).sort_values("f1")

plt.figure(figsize=(10, 12))
colors = ["#d9534f" if f < 0.70 else "#5bc0de" if f < 0.85 else "#5cb85c"
          for f in f1_df["f1"]]
plt.barh(f1_df["label"], f1_df["f1"], color=colors)
plt.axvline(0.70, color="red",   linestyle="--", linewidth=1, label="0.70 threshold")
plt.axvline(0.85, color="green", linestyle="--", linewidth=1, label="0.85 threshold")
plt.xlabel("F1 Score")
plt.title("Per-Label F1 — Test Set  (VulnDetector v3)", fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig("f1_per_label_v3.png", dpi=150)
plt.show()


# ============================================================
# 14.  SAVE ARTEFACTS
# ============================================================

model.save(CFG["model_path"])

with open(CFG["tok_path"], "wb") as f:
    pickle.dump(tokenizer, f)

with open(CFG["enc_path"], "wb") as f:
    pickle.dump(encoder, f)

print(f"\nSaved  →  {CFG['model_path']}  |  {CFG['tok_path']}  |  {CFG['enc_path']}")


# ============================================================
# 15.  PREDICTION FUNCTION
#      Now returns cwe_id + owasp_category alongside severity.
# ============================================================

def predict_vulnerability(code_snippet: str, threshold: float = 0.50) -> dict:
    cleaned  = clean_code(code_snippet)
    seq      = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(
        seq, maxlen=CFG["max_len"], padding="post", truncating="post"
    )
    probs    = model.predict(padded, verbose=0)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top_lbl  = encoder.inverse_transform([top3_idx[0]])[0]
    top_conf = float(probs[top3_idx[0]])
    top3     = [
        (encoder.inverse_transform([i])[0], round(float(probs[i]) * 100, 2))
        for i in top3_idx
    ]
    return {
        "prediction"   : top_lbl,
        "severity"     : SEVERITY.get(top_lbl, "UNKNOWN"),
        "cwe_id"       : CWE_MAP.get(top_lbl,  "N/A"),
        "owasp"        : OWASP.get(top_lbl,     "N/A"),
        "confidence"   : round(top_conf * 100, 2),
        "top3"         : top3,
        "is_uncertain" : top_conf < threshold,
    }


# ============================================================
# 16.  INTERACTIVE CONSOLE
# ============================================================

VALID_LABELS = sorted(LABEL_META.keys())

print("\n" + "=" * 54)
print("  AI Vulnerability Detection Console  v3.0")
print("  Dataset: 25 124 entries  |  35 classes")
print("=" * 54)
print("  exit              — quit")
print("  labels            — list all 35 vulnerability classes")
print("  feedback <LABEL>  — correct last prediction")
print("  retrain           — fine-tune on buffered feedback")
print("=" * 54)

feedback_buffer = []
last_snippet    = None

while True:
    print()
    user_input = input("CODE >>> ").strip()
    if not user_input:
        continue

    # ── Commands ──────────────────────────────────────────
    if user_input.lower() == "exit":
        print("Goodbye.")
        break

    if user_input.lower() == "labels":
        for i, lbl in enumerate(VALID_LABELS, 1):
            meta = LABEL_META[lbl]
            print(f"  {i:>2}. {lbl:<30} {meta[0]:<10}  {meta[1]}")
        continue

    if user_input.lower() == "retrain":
        if not feedback_buffer:
            print("No feedback yet.")
            continue
        fb_df = pd.DataFrame(feedback_buffer)
        fb_X  = pad_sequences(
            tokenizer.texts_to_sequences(
                [clean_code(c) for c in fb_df["code"].values]
            ),
            maxlen=CFG["max_len"], padding="post", truncating="post"
        )
        fb_y = tf.keras.utils.to_categorical(
            encoder.transform(fb_df["label"].values),
            num_classes=n_classes,
        )
        model.fit(fb_X, fb_y, epochs=5, batch_size=16, verbose=1)
        model.save(CFG["model_path"])
        print(f"Fine-tune done on {len(fb_df)} samples. Model saved.")
        continue

    if user_input.lower().startswith("feedback "):
        corrected = user_input.split(maxsplit=1)[1].strip().upper()
        if last_snippet and corrected in VALID_LABELS:
            feedback_buffer.append({"code": last_snippet, "label": corrected})
            print(f"  Saved feedback → {corrected}  (buffer: {len(feedback_buffer)})")
        else:
            print(f"  Unknown label. Type 'labels' to see all {len(VALID_LABELS)} classes.")
        continue

    # ── Prediction ────────────────────────────────────────
    last_snippet = user_input
    r    = predict_vulnerability(user_input)
    flag = "⚠  UNCERTAIN" if r["is_uncertain"] else "✓  CONFIDENT"

    print(f"\n{'─'*54}")
    print(f"  Prediction  : {r['prediction']:<28} [{flag}]")
    print(f"  Severity    : {r['severity']}")
    print(f"  CWE         : {r['cwe_id']}")
    print(f"  OWASP       : {r['owasp']}")
    print(f"  Confidence  : {r['confidence']}%")
    print(f"  Top 3:")
    for lbl, pct in r["top3"]:
        bar = "█" * int(pct / 4)
        print(f"    {lbl:<30} {pct:6.2f}%  {bar}")
    print(f"{'─'*54}")

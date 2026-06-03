# ╔══════════════════════════════════════════════════════════════╗
# ║   AI-SAST VulnDetector v4.0 — Enhanced Colab Training       ║
# ║   Key upgrades:                                              ║
# ║   • Longer context window (max_len 32→128)                  ║
# ║   • Focal loss — crushes false positives                    ║
# ║   • Context-aware tokeniser (preserves \n as token)         ║
# ║   • Deeper BiLSTM + wider attention                         ║
# ║   • Hard negative SAFE examples in dataset                  ║
# ╚══════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────
# CELL 1 — Setup
# ─────────────────────────────────────────────────────────────────
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_MODULE_LOADING"]   = "LAZY"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore")

import json, re, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers, Model, Input, backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
)
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

print(f"TensorFlow : {tf.__version__}")
print(f"GPU        : {tf.config.list_physical_devices('GPU')}")
print("✓ Ready")


# ─────────────────────────────────────────────────────────────────
# CELL 2 — Upload dataset
# ─────────────────────────────────────────────────────────────────
# Option A: Upload sast_training_v2.json from local machine
# from google.colab import files
# uploaded = files.upload()
# DATASET_PATH = list(uploaded.keys())[0]

# Option B: Mount Drive
# from google.colab import drive
# drive.mount('/content/drive')
# DATASET_PATH = '/content/drive/MyDrive/sast_training_v2.json'

# Option C: wget from GitHub
# !wget -q https://raw.githubusercontent.com/gijutsu-hub/aisastscanner/main/sast_training_v2.json
# DATASET_PATH = 'sast_training_v2.json'

DATASET_PATH = "sast_training_v2.json"


# ─────────────────────────────────────────────────────────────────
# CELL 3 — Config
# ─────────────────────────────────────────────────────────────────
CFG = {
    "json_file"    : DATASET_PATH,
    "test_size"    : 0.15,
    "val_size"     : 0.15,

    # ── Tokeniser ─────────────────────────────────────────
    # Longer window captures multi-line function context
    "vocab_size"   : 8_000,
    "max_len"      : 128,      # ← 32→128: captures full functions

    # ── Embedding ─────────────────────────────────────────
    "embed_dim"    : 128,
    "embed_dropout": 0.20,

    # ── CNN branch ────────────────────────────────────────
    "cnn_filters"  : 192,
    "cnn_kernels"  : [2, 3, 5, 7],   # ← added kernel 7 for longer patterns

    # ── BiLSTM ────────────────────────────────────────────
    "lstm_units"   : 192,      # ← 128→192
    "lstm_layers"  : 2,        # ← stacked BiLSTM

    # ── Multi-Head Attention ───────────────────────────────
    "attn_heads"   : 8,        # ← 4→8
    "attn_key_dim" : 64,       # ← 32→64

    # ── Dense head ────────────────────────────────────────
    "dense_units"  : [512, 256, 128],
    "dropout_rate" : 0.45,
    "l2_reg"       : 1e-4,

    # ── Training ──────────────────────────────────────────
    "batch_size"   : 128,
    "epochs"       : 80,
    "lr"           : 8e-4,
    "label_smooth" : 0.03,     # ← lighter: model needs to commit on SAFE

    # Focal loss gamma — higher = more focus on hard misclassified examples
    "focal_gamma"  : 2.0,

    # ── Artefacts ─────────────────────────────────────────
    "model_path"   : "vuln_model_v4.keras",
    "tok_path"     : "tokenizer_v4.pkl",
    "enc_path"     : "label_encoder_v4.pkl",
}

print("Config:")
for k, v in CFG.items():
    print(f"  {k:<18} : {v}")


# ─────────────────────────────────────────────────────────────────
# CELL 4 — Load dataset
# ─────────────────────────────────────────────────────────────────
with open(CFG["json_file"]) as f:
    raw = json.load(f)

df = (pd.DataFrame(raw)
        .dropna(subset=["code", "label"])
        .drop_duplicates(subset=["code"]))

print(f"\n{'='*52}")
print(f"  Total   : {len(df):,}")
print(f"  Labels  : {df['label'].nunique()}")
print(f"{'='*52}")
print(df["label"].value_counts().to_string())

# Sanity check — warn if dataset looks too small
if len(df) < 10_000:
    print(f"""
╔══════════════════════════════════════════════════════╗
║  WARNING: Only {len(df):,} entries found.                   ║
║  Expected ~25,566 (sast_training_v2.json).           ║
║  You may have uploaded the wrong file.               ║
║                                                      ║
║  Correct file  : sast_training_v2.json  (~7.5 MB)   ║
║  Wrong file    : training.json          (~900 KB)    ║
╚══════════════════════════════════════════════════════╝
    """)
else:
    print(f"  ✓ Dataset size looks correct ({len(df):,} entries)")

if "context" in df.columns:
    print(f"\nContext breakdown:")
    print(df["context"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────────
# CELL 5 — Pre-processing
#   Key change: preserve \n as a special token <NL> so the model
#   knows where function boundaries are.
# ─────────────────────────────────────────────────────────────────
def clean_code(snippet: str) -> str:
    # Replace newlines with a special token BEFORE lowering
    snippet = snippet.replace("\r\n", " <NL> ").replace("\n", " <NL> ")
    snippet = snippet.lower().strip()
    # Space-pad operators/delimiters
    snippet = re.sub(r"([(){}\[\];:,.<>+\-*/=!&|^~%@#$])", r" \1 ", snippet)
    snippet = re.sub(r"\s+", " ", snippet)
    return snippet

df["code_clean"] = df["code"].apply(clean_code)
codes  = df["code_clean"].values
labels = df["label"].values

print("Sample cleaned:")
for i in range(3):
    print(f"\n  [{labels[i]}] {codes[i][:120]}")


# ─────────────────────────────────────────────────────────────────
# CELL 6 — Encode labels
# ─────────────────────────────────────────────────────────────────
encoder   = LabelEncoder()
y_int     = encoder.fit_transform(labels)
n_classes = len(encoder.classes_)
y_onehot  = tf.keras.utils.to_categorical(y_int, num_classes=n_classes)

print(f"\n{n_classes} classes: {list(encoder.classes_)}")


# ─────────────────────────────────────────────────────────────────
# CELL 7 — Tokenise
# ─────────────────────────────────────────────────────────────────
tokenizer = Tokenizer(
    num_words = CFG["vocab_size"],
    oov_token = "<OOV>",
    filters   = "",
    lower     = False,
)
tokenizer.fit_on_texts(codes)

X = tokenizer.texts_to_sequences(codes)
X = pad_sequences(X, maxlen=CFG["max_len"], padding="post", truncating="post")

actual_vocab = min(len(tokenizer.word_index)+1, CFG["vocab_size"])
print(f"Vocab  : {actual_vocab:,}")
print(f"X      : {X.shape}")

# Check <NL> token was captured
nl_id = tokenizer.word_index.get("<nl>", "NOT FOUND")
print(f"<NL> token id : {nl_id}")


# ─────────────────────────────────────────────────────────────────
# CELL 8 — Stratified split
# ─────────────────────────────────────────────────────────────────
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
print(f"Train {len(X_train):,}  |  Val {len(X_val):,}  |  Test {len(X_test):,}")

cw_values     = compute_class_weight("balanced",
                    classes=np.unique(y_train_int), y=y_train_int)
class_weights = dict(enumerate(cw_values))
print(f"Class weights computed for {len(class_weights)} classes ✓")


# ─────────────────────────────────────────────────────────────────
# CELL 9 — Focal Loss
#   Focal loss penalises easy examples less and hard examples more.
#   This directly addresses false positives: the model gets more
#   gradient signal from misclassified SAFE vs VULNERABLE cases.
# ─────────────────────────────────────────────────────────────────
# Compatible decorator across TF 2.x versions
try:
    _reg = tf.keras.saving.register_keras_serializable(package="sast")
except AttributeError:
    try:
        _reg = tf.keras.utils.register_keras_serializable(package="sast")
    except AttributeError:
        # Fallback: no-op decorator — compile=False handles it in scanner
        def _reg(cls): return cls

@_reg
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal loss = -(1 - pt)^gamma * log(pt)
    Registered with @register_keras_serializable so the saved .keras
    model can be loaded with load_model() without compile=False.
    gamma=0 → standard cross-entropy
    gamma=2 → strong focus on hard misclassified examples (default)
    """
    def __init__(self, gamma=2.0, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        n_cls  = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / n_cls
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce     = -y_true * tf.math.log(y_pred)
        pt     = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        fl     = tf.pow(1.0 - pt, self.gamma) * ce
        return tf.reduce_mean(tf.reduce_sum(fl, axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma,
                    "label_smoothing": self.label_smoothing})
        return cfg


def focal_loss(gamma=2.0, label_smoothing=0.0):
    """Convenience wrapper — returns a FocalLoss instance."""
    return FocalLoss(gamma=gamma, label_smoothing=label_smoothing)

print("Focal loss defined ✓")


# ─────────────────────────────────────────────────────────────────
# CELL 10 — Build model v4
#   Changes vs v3:
#   • max_len 32 → 128 (captures full functions)
#   • Stacked BiLSTM (2 layers)
#   • 8-head attention with larger key dim
#   • Added kernel size 7 in CNN branch
#   • Focal loss instead of cross-entropy
# ─────────────────────────────────────────────────────────────────
def build_model(vocab_size, embed_dim, max_len, n_classes, cfg):

    inp = Input(shape=(max_len,), name="input")

    # ── Embedding ─────────────────────────────────────────
    emb = layers.Embedding(vocab_size, embed_dim, name="emb")(inp)
    emb = layers.SpatialDropout1D(cfg["embed_dropout"], name="emb_drop")(emb)

    # ── CNN branch — multi-scale local patterns ───────────
    pools = []
    for k in cfg["cnn_kernels"]:
        x = layers.Conv1D(
            cfg["cnn_filters"], k,
            padding="same", activation="relu",
            kernel_regularizer=l2(cfg["l2_reg"]),
            name=f"conv_{k}",
        )(emb)
        x = layers.GlobalMaxPooling1D(name=f"pool_{k}")(x)
        pools.append(x)

    cnn = layers.Concatenate(name="cnn_cat")(pools)
    cnn = layers.Dense(256, activation="relu", name="cnn_fc")(cnn)
    cnn = layers.BatchNormalization(name="cnn_bn")(cnn)
    cnn = layers.Dropout(cfg["dropout_rate"], name="cnn_drop")(cnn)

    # ── Stacked BiLSTM ────────────────────────────────────
    lstm = layers.Bidirectional(
        layers.LSTM(cfg["lstm_units"], return_sequences=True, dropout=0.2),
        name="bilstm_1",
    )(emb)
    lstm = layers.Bidirectional(
        layers.LSTM(cfg["lstm_units"] // 2, return_sequences=True, dropout=0.1),
        name="bilstm_2",
    )(lstm)

    # ── Multi-Head Attention + residual ───────────────────
    attn, _ = layers.MultiHeadAttention(
        num_heads = cfg["attn_heads"],
        key_dim   = cfg["attn_key_dim"],
        name      = "mha",
    )(lstm, lstm, return_attention_scores=True)

    # Residual: project lstm to match attn dim if needed
    attn = layers.Add(name="residual")([lstm, attn])
    attn = layers.LayerNormalization(name="ln")(attn)
    seq  = layers.GlobalAveragePooling1D(name="gap")(attn)
    seq  = layers.Dense(256, activation="relu", name="seq_fc")(seq)
    seq  = layers.BatchNormalization(name="seq_bn")(seq)
    seq  = layers.Dropout(cfg["dropout_rate"], name="seq_drop")(seq)

    # ── Merge → Dense head ────────────────────────────────
    x = layers.Concatenate(name="merge")([cnn, seq])

    for i, units in enumerate(cfg["dense_units"]):
        x = layers.Dense(
            units, activation="relu",
            kernel_regularizer=l2(cfg["l2_reg"]),
            name=f"fc_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(cfg["dropout_rate"], name=f"drop_{i}")(x)

    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    return Model(inp, out, name="VulnDetector_v4")


model = build_model(
    vocab_size = CFG["vocab_size"],
    embed_dim  = CFG["embed_dim"],
    max_len    = CFG["max_len"],
    n_classes  = n_classes,
    cfg        = CFG,
)
model.summary()
print(f"\nTotal params : {model.count_params():,}")


# ─────────────────────────────────────────────────────────────────
# CELL 11 — Compile with focal loss
# ─────────────────────────────────────────────────────────────────
model.compile(
    optimizer = tf.keras.optimizers.Adam(
        learning_rate = CFG["lr"],
        clipnorm      = 1.0,
    ),
    loss    = focal_loss(
        gamma          = CFG["focal_gamma"],
        label_smoothing= CFG["label_smooth"],
    ),
    metrics = ["accuracy"],
)
print("Compiled with Focal Loss ✓")


# ─────────────────────────────────────────────────────────────────
# CELL 12 — Callbacks + false-positive tracker
# ─────────────────────────────────────────────────────────────────

# Track false positive rate on validation set each epoch
safe_idx = list(encoder.classes_).index("SAFE")

class FPRCallback(tf.keras.callbacks.Callback):
    """Prints False Positive Rate (SAFE predicted as VULN) each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        preds   = np.argmax(self.model.predict(X_val, verbose=0), axis=1)
        true    = y_val_int
        safe_mask   = true == safe_idx
        fp_rate = np.mean(preds[safe_mask] != safe_idx) if safe_mask.any() else 0.0
        fn_rate = np.mean(preds[~safe_mask] == safe_idx) if (~safe_mask).any() else 0.0
        print(f"  epoch {epoch+1:>3}  FP-rate: {fp_rate*100:.1f}%  "
              f"FN-rate: {fn_rate*100:.1f}%")

callbacks = [
    EarlyStopping(
        monitor              = "val_accuracy",
        patience             = 12,
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
        patience = 5,
        min_lr   = 1e-6,
        verbose  = 1,
    ),
    FPRCallback(),
]
print("Callbacks ready ✓")


# ─────────────────────────────────────────────────────────────────
# CELL 13 — Train 🚀
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*52)
print("  Training VulnDetector v4 ...")
print("  • Focal loss  (γ=2.0)  — hard example focus")
print("  • max_len=128          — captures full functions")
print("  • Stacked BiLSTM       — deeper sequential context")
print("  • 8-head attention     — wider context alignment")
print("="*52 + "\n")

history = model.fit(
    X_train, y_train,
    validation_data = (X_val, y_val),
    epochs          = CFG["epochs"],
    batch_size      = CFG["batch_size"],
    class_weight    = class_weights,
    callbacks       = callbacks,
    verbose         = 1,
)


# ─────────────────────────────────────────────────────────────────
# CELL 14 — Evaluate with FP/FN breakdown
# ─────────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test_oh, axis=1)

print(f"\n{'='*52}")
print(f"  Test Accuracy : {test_acc*100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")
print(f"{'='*52}")

# FP / FN breakdown for SAFE class
safe_mask = y_true == safe_idx
fp        = np.sum((y_pred != safe_idx) & safe_mask)   # SAFE predicted as VULN
fn        = np.sum((y_pred == safe_idx) & ~safe_mask)  # VULN predicted as SAFE
tp_safe   = np.sum((y_pred == safe_idx) & safe_mask)

print(f"\n  SAFE class breakdown:")
print(f"    True Positives  (correctly SAFE)  : {tp_safe}")
print(f"    False Positives (SAFE→VULN)       : {fp}   ← want this low")
print(f"    False Negatives (VULN→SAFE)       : {fn}   ← want this low")
if safe_mask.sum() > 0:
    print(f"    FP rate                          : {fp/safe_mask.sum()*100:.1f}%")

print("\nFull Classification Report:")
print(classification_report(y_true, y_pred,
      target_names=encoder.classes_, digits=4))


# ─────────────────────────────────────────────────────────────────
# CELL 15 — Plots
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14,5))
for ax, (tk, vk), title in zip(
    axes,
    [("accuracy","val_accuracy"),("loss","val_loss")],
    ["Accuracy","Loss"]
):
    ax.plot(history.history[tk], label="Train", linewidth=2)
    ax.plot(history.history[vk], label="Val",   linewidth=2)
    ax.set_title(f"{title} — VulnDetector v4", fontsize=13)
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves_v4.png", dpi=150)
plt.show()

# F1 per label
f1_scores = f1_score(y_true, y_pred, average=None)
f1_df = pd.DataFrame({"label":encoder.classes_,"f1":f1_scores}).sort_values("f1")
plt.figure(figsize=(10,14))
colors = ["#e74c3c" if f<0.70 else "#5bc0de" if f<0.85 else "#5cb85c"
          for f in f1_df["f1"]]
plt.barh(f1_df["label"], f1_df["f1"], color=colors)
plt.axvline(0.70, color="red",   linestyle="--", linewidth=1, label="0.70")
plt.axvline(0.85, color="green", linestyle="--", linewidth=1, label="0.85")
plt.xlabel("F1 Score")
plt.title("Per-Label F1 — VulnDetector v4")
plt.legend(); plt.tight_layout()
plt.savefig("f1_per_label_v4.png", dpi=150)
plt.show()

# Confusion matrix (35x35)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(22,18))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_,
            annot_kws={"size":7})
plt.title("Confusion Matrix — VulnDetector v4")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0,  fontsize=8)
plt.tight_layout()
plt.savefig("confusion_matrix_v4.png", dpi=150)
plt.show()


# ─────────────────────────────────────────────────────────────────
# CELL 16 — Save + download
# ─────────────────────────────────────────────────────────────────
model.save(CFG["model_path"])
with open(CFG["tok_path"], "wb") as f: pickle.dump(tokenizer, f)
with open(CFG["enc_path"], "wb") as f: pickle.dump(encoder, f)

print(f"\n✓  Model     → {CFG['model_path']}")
print(f"✓  Tokenizer → {CFG['tok_path']}")
print(f"✓  Encoder   → {CFG['enc_path']}")

try:
    from google.colab import files
    for fname in [
        CFG["model_path"], CFG["tok_path"], CFG["enc_path"],
        "training_curves_v4.png", "f1_per_label_v4.png",
    ]:
        files.download(fname)
        print(f"  Downloaded → {fname}")
except ImportError:
    print("[Not in Colab] Copy files manually.")


# ─────────────────────────────────────────────────────────────────
# CELL 17 — Quick test with cross-function examples
# ─────────────────────────────────────────────────────────────────
LABEL_META = {
    "SQLI":("CRITICAL","CWE-89"),"XSS":("HIGH","CWE-79"),
    "COMMAND_INJECTION":("CRITICAL","CWE-78"),"RCE":("CRITICAL","CWE-94"),
    "PATH_TRAVERSAL":("HIGH","CWE-22"),"SSRF":("HIGH","CWE-918"),
    "SSTI":("CRITICAL","CWE-1336"),"SAFE":("NONE","N/A"),
}

def predict(snippet, threshold=0.50):
    cleaned  = clean_code(snippet)
    seq      = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(seq, maxlen=CFG["max_len"],
                             padding="post", truncating="post")
    probs    = model.predict(padded, verbose=0)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top_lbl  = encoder.inverse_transform([top3_idx[0]])[0]
    top_conf = float(probs[top3_idx[0]])
    meta     = LABEL_META.get(top_lbl, ("UNKNOWN","N/A"))
    return {
        "prediction" : top_lbl,
        "severity"   : meta[0],
        "cwe_id"     : meta[1],
        "confidence" : round(top_conf*100, 2),
        "uncertain"  : top_conf < threshold,
        "top3"       : [(encoder.inverse_transform([i])[0],
                         round(float(probs[i])*100,2)) for i in top3_idx],
    }

TEST_CASES = [
    # Should be SAFE
    ("cursor.execute('SELECT * FROM users WHERE id=%s', (uid,))", "SAFE"),
    ("def get_safe(f):\n    p=os.path.realpath(os.path.join(BASE,f))\n    assert p.startswith(BASE)\n    return open(p).read()", "SAFE"),
    ("const hash = await bcrypt.hash(password, 12)", "SAFE"),

    # Should be VULNERABLE
    ("query='SELECT * FROM users WHERE id='+user_id; db.execute(query)", "SQLI"),
    ("def get_u(): return request.args.get('url')\ndef fetch(u): return requests.get(u)\nfetch(get_u())", "SSRF"),
    ("return render_template_string(request.form['tmpl'])", "SSTI"),
    ("document.getElementById('out').innerHTML = location.search", "XSS"),
]

print(f"\n{'='*60}")
print("  Cross-function prediction test")
print(f"{'='*60}")
correct = 0
for snippet, expected in TEST_CASES:
    r = predict(snippet)
    ok = "✓" if r["prediction"] == expected else "✗"
    if r["prediction"] == expected: correct += 1
    print(f"\n  {ok} Expected: {expected:<25} Got: {r['prediction']}")
    print(f"     Confidence: {r['confidence']:.1f}%  | {snippet[:60]}")

print(f"\n  Score: {correct}/{len(TEST_CASES)}")

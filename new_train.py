# train_model.py
import re
import time
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# =========================
# Config
# =========================
CSV_PATH = "improved_diversed_data.csv"
MODEL_NAME = "tbs17/MathBERT"
MAX_LEN = 320                   # keep formulas intact; trim context
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL  = 32
LR = 2e-5
EPOCHS = 10
PATIENCE = 2
PRINT_EVERY = 10
KEEP_CTX_CHARS = 240
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load & labels
# =========================
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])
num_classes = len(le.classes_)
print("Labels:", list(le.classes_))

# =========================
# Tokenizer (+ special tokens)
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
special_ops = ["[MATH]", "[PROB]", "[ANS]", "ADD","SUB","MUL","DIV","EQ","SUP","SUBS","FRAC","SQRT","SUM","INT","REL","BR"]
tokenizer.add_special_tokens({"additional_special_tokens": special_ops})

# =========================
# Math helpers
# =========================
SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "^{0}", "¹": "^{1}", "²": "^{2}", "³": "^{3}", "⁴": "^{4}",
    "⁵": "^{5}", "⁶": "^{6}", "⁷": "^{7}", "⁸": "^{8}", "⁹": "^{9}",
})
GREEK_MAP = {
    "α":"\\alpha","β":"\\beta","γ":"\\gamma","δ":"\\delta","ε":"\\epsilon","ζ":"\\zeta","η":"\\eta","θ":"\\theta",
    "ι":"\\iota","κ":"\\kappa","λ":"\\lambda","μ":"\\mu","ν":"\\nu","ξ":"\\xi","ο":"o","π":"\\pi","ρ":"\\rho",
    "σ":"\\sigma","τ":"\\tau","υ":"\\upsilon","φ":"\\phi","χ":"\\chi","ψ":"\\psi","ω":"\\omega",
    "Α":"A","Β":"B","Γ":"\\Gamma","Δ":"\\Delta","Ε":"E","Ζ":"Z","Η":"H","Θ":"\\Theta","Ι":"I","Κ":"K","Λ":"\\Lambda",
    "Μ":"M","Ν":"N","Ξ":"\\Xi","Ο":"O","Π":"\\Pi","Ρ":"P","Σ":"\\Sigma","Τ":"T","Υ":"\\Upsilon","Φ":"\\Phi","Χ":"X",
    "Ψ":"\\Psi","Ω":"\\Omega"
}
OP_MAP = {
    "×": "\\cdot", "∙": "\\cdot", "•":"\\cdot", "·":"\\cdot",
    "÷": "\\frac",
    "±": "\\pm", "∓": "\\mp",
    "≤": "\\le", "≥": "\\ge", "≠":"\\ne", "≈":"\\approx",
    "→":"\\to", "⇒":"\\Rightarrow", "←":"\\leftarrow", "⇐":"\\Leftarrow",
    "∞":"\\infty", "√":"\\sqrt", "∑":"\\sum", "∏":"\\prod", "∫":"\\int",
}
LATEX_SPANS = [
    re.compile(r"\$(.+?)\$", re.DOTALL),
    re.compile(r"\\\((.+?)\\\)", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
]
_SIMPLE_FRAC = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")

def _unicode_math_to_latex(s: str) -> str:
    s = s.translate(SUPERSCRIPT_MAP)
    for k, v in GREEK_MAP.items(): s = s.replace(k, v)
    for k, v in OP_MAP.items():    s = s.replace(k, v)
    s = re.sub(r"\b[xX]\b", lambda m: r"\cdot", s)
    return s

def _first_latex(text: str):
    for pat in LATEX_SPANS:
        m = pat.search(text)
        if m: return m.group(1).strip()
    return None

def _heuristic_math(text: str) -> str:
    m = re.search(r"[=+\-*/^]|\\frac|\\sqrt|≤|≥|≠|≈|∑|∫|π|θ|α|β|√|\b[xX]\b", text)
    if not m: return _unicode_math_to_latex(text[:40]).strip()
    i = m.start(); L = max(0, i-20); R = min(len(text), i+20)
    return _unicode_math_to_latex(text[L:R]).strip()

def _cleanup_math(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s*\|\s*$", "", s)
    return s

def _normalize_mixed_number(s: str) -> str:
    m = re.match(r"^\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s*$", s)
    if not m: return s
    whole, num, den = m.groups()
    return f"{whole} \\frac{{{num}}}{{{den}}}"

def _to_frac_latex(s: str) -> str:
    # Convert simple a/b to \frac{a}{b}; safe if already LaTeX
    return _SIMPLE_FRAC.sub(lambda m: f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}", s)

def _inject_math_marker(ctx: str) -> str:
    # Put [MATH] before first math-like token (digit or backslash)
    m = re.search(r"[0-9\\]", ctx)
    if not m: return "[MATH] " + ctx
    i = m.start()
    return ctx[:i] + "[MATH] " + ctx[i:]

def _opt_ops(latex: str) -> str:
    ops = []
    i = 0
    while i < len(latex):
        if latex.startswith("\\frac", i): ops.append("FRAC"); i += 5; continue
        if latex.startswith("\\sqrt", i): ops.append("SQRT"); i += 5; continue
        if latex.startswith("\\sum", i):  ops.append("SUM");  i += 4; continue
        if latex.startswith("\\int", i):  ops.append("INT");  i += 4; continue
        if latex.startswith("\\cdot", i): ops.append("MUL");  i += 5; continue
        if latex.startswith("\\le", i) or latex.startswith("\\ge", i) or latex.startswith("\\ne", i):
            ops.append("REL"); i += 3; continue
        ch = latex[i]
        if   ch == "+": ops.append("ADD")
        elif ch == "-": ops.append("SUB")
        elif ch == "*": ops.append("MUL")
        elif ch == "/": ops.append("DIV")
        elif ch == "=": ops.append("EQ")
        elif ch == "^": ops.append("SUP")
        elif ch == "_": ops.append("SUBS")
        elif ch in "()[]{}": ops.append("BR")
        i += 1
    compact = []
    for o in ops:
        if not compact or compact[-1] != o:
            compact.append(o)
    return " ".join(compact) if compact else "CONST"

def _extract_problem_and_answer(raw: str):
    text = str(raw)
    m = re.search(r"(?i)\bstudent\s*answer\s*:\s*", text)
    if m:
        prob = text[:m.start()].replace("Problem:", "").strip()
        ans  = text[m.end():].strip()
    else:
        prob, ans = text.replace("Problem:", "").strip(), ""
    p_l = _first_latex(prob) or _heuristic_math(prob)
    a_l = _first_latex(ans)  or (_heuristic_math(ans) if ans else "")

    p_l = _cleanup_math(_normalize_mixed_number(p_l))
    a_l = _cleanup_math(_normalize_mixed_number(a_l))
    p_l = _to_frac_latex(p_l)
    a_l = _to_frac_latex(a_l)

    ctx = f"[PROB] {prob} [ANS] {ans}"
    ctx = _inject_math_marker(ctx)
    if len(ctx) > KEEP_CTX_CHARS: ctx = ctx[:KEEP_CTX_CHARS]
    return p_l, a_l, ctx

# =========================
# Dataset
# =========================
class ErrorDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.texts = df["input"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        raw = self.texts[idx]
        prob_latex, ans_latex, ctx = _extract_problem_and_answer(raw)

        # Segment A: formulas
        segA = f"[PROB] {prob_latex} [ANS] {ans_latex}".strip()
        # Segment B: context + operator tags from both formulas
        optA = _opt_ops(prob_latex) if prob_latex else "CONST"
        optB = _opt_ops(ans_latex)  if ans_latex  else "CONST"
        segB = f"{ctx} {optA} {optB}".strip()   # ← include operator tags

        enc = self.tokenizer(
            text=segA,
            text_pair=segB,
            padding="max_length",
            truncation="only_second",          # keep formulas intact; trim context
            max_length=self.max_len,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            # optional for debugging
            "raw": raw, "prob_latex": prob_latex, "ans_latex": ans_latex, "ctx": ctx,
            "special_mask": enc["special_tokens_mask"].squeeze(0)
        }

# =========================
# Split & loaders
# =========================
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['label'])
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['label'])

train_dataset = ErrorDataset(train_df, tokenizer, max_len=MAX_LEN)
val_dataset   = ErrorDataset(val_df, tokenizer,   max_len=MAX_LEN)
test_dataset  = ErrorDataset(test_df, tokenizer,  max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE_EVAL,  shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE_EVAL,  shuffle=False, num_workers=0, pin_memory=False)

# quick sanity peek
if len(train_dataset) > 0:
    s = train_dataset[0]
    print("\nSANITY SAMPLE")
    print(" RAW  :", s["raw"][:120])
    print(" PROB :", s["prob_latex"])
    print(" ANS  :", s["ans_latex"])
    print(" CTX  :", s["ctx"][:120])
    # Verify two segments exist (should contain {0,1})
    tti_set = set(s["token_type_ids"].tolist())
    print("SEGMENT IDS present:", tti_set)

# =========================
# Model (passes token_type_ids)
# =========================
class MathBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes, tokenizer_len=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if tokenizer_len is not None:
            self.bert.resize_token_embeddings(tokenizer_len)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

model = MathBERTClassifier(MODEL_NAME, num_classes, tokenizer_len=len(tokenizer)).to(device)

# =========================
# Loss, Optimizer
# =========================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="sum"):  # sum → proper running avg
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, target):
        ce = self.ce(logits, target)           # [B]
        p = torch.exp(-ce)
        loss = ((1 - p) ** self.gamma) * ce    # [B]
        return loss.sum() if self.reduction == "sum" else loss.mean()

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df["label_id"]),
    y=train_df["label_id"]
)
weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)
loss_fn = FocalLoss(weight=weights_tensor, reduction="sum")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# =========================
# Train / Validate / Test
# =========================
best_val_f1 = -1.0
best_state = None
early_stop = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss_sum = 0.0
    running_correct  = 0
    running_seen     = 0
    start_t = time.time()

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = loss_fn(logits, labels)   # summed over batch

        # ---- metrics before step (cumulative per-example) ----
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        bs = labels.size(0)

        running_loss_sum += loss.item()          # sum over examples
        running_correct  += correct
        running_seen     += bs

        batch_acc = correct / bs
        running_loss_avg = running_loss_sum / max(1, running_seen)
        running_acc = running_correct / max(1, running_seen)

        # ---- optimize ----
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % PRINT_EVERY == 0:
            print(f"Epoch {epoch:02d} | Batch {step:04d}/{len(train_loader):04d} "
                  f"| Loss: {(loss.item()/bs):.4f} "
                  f"| BatchAcc: {batch_acc:.4f} "
                  f"| RunningLoss: {running_loss_avg:.4f} "
                  f"| RunningAcc: {running_acc:.4f}")

    train_loss_epoch = running_loss_sum / max(1, running_seen)
    train_acc_epoch  = running_acc if running_seen > 0 else 0.0

    # ---------- Validation ----------
    model.eval()
    val_loss_sum = 0.0
    val_seen     = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(logits, labels)      # summed

            val_loss_sum += loss.item()
            val_seen     += labels.size(0)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_loss_sum / max(1, val_seen)
    val_acc  = accuracy_score(all_labels, all_preds)
    val_f1   = f1_score(all_labels, all_preds, average="macro")

    dur = time.time() - start_t
    print(f"\n=== Epoch {epoch}/{EPOCHS} ({dur:.1f}s) ===")
    print(f"Train: loss={train_loss_epoch:.4f} | acc={train_acc_epoch:.4f}")
    print(f"Valid: loss={val_loss:.4f} | acc={val_acc:.4f} | macroF1={val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = model.state_dict()
        early_stop = 0
        torch.save(best_state, "best_mathbert_model.pt")
        print("  ✓ Best model saved.")
    else:
        early_stop += 1
        if early_stop >= PATIENCE:
            print("  ↳ Early stopping.")
            break

# ---------- Test on best ----------
print("\nLoading best checkpoint for testing...")
if best_state is None:
    best_state = model.state_dict()
torch.save(best_state, "best_mathbert_model.pt")
model.load_state_dict(torch.load("best_mathbert_model.pt", map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

print("\nConfusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Saved confusion_matrix.png")
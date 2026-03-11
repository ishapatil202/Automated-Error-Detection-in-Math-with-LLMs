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

# 1. Load cleaned dataset (no 'no_error')
df = pd.read_csv("improved_diversed_data.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the data

# 2. Encode labels
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])
num_classes = len(le.classes_)

# 3. Tokenizer
model_name = "tbs17/MathBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 4. Dataset class
class ErrorDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["input"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx])
        }

# 5. Split dataset (80/10/10 train/val/test, stratified)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

# Save test set for use in prediction script
test_df.to_csv("test_set.csv", index=False)

train_dataset = ErrorDataset(train_df, tokenizer)
val_dataset = ErrorDataset(val_df, tokenizer)
test_dataset = ErrorDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# 6. Define model
class MathBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

model = MathBERTClassifier(model_name, num_classes)

# 7. Focal Loss with class weights
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, input, target):
        logp = self.ce(input, target)   # scalar loss per batch (since CE reduced)
        p = torch.exp(-logp)
        return ((1 - p) ** self.gamma * logp).mean()

# 8. Optimizer & Device
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# (small safety: put class weights on the same device)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df["label_id"]), y=df["label_id"])
weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)
loss_fn = FocalLoss(weight=weights_tensor)

# 9. Training loop with early stopping and confusion matrix image
best_val_f1 = 0.0
best_model_state = None
early_stop_counter = 0
patience = 2

for epoch in range(10):
    total_loss = 0.0
    model.train()

    # NEW: running accuracy/loss trackers for the epoch
    running_correct = 0
    running_seen = 0

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # NEW: compute batch accuracy and running stats
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum().item()
        batch_size = labels.size(0)
        running_correct += correct
        running_seen += batch_size

        batch_acc = correct / batch_size
        running_loss_avg = total_loss / running_seen
        running_acc = running_correct / running_seen

        # ORIGINAL print cadence kept, with richer info
        if i % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {i} "
                  f"| Loss: {loss.item():.4f} "
                  f"| BatchAcc: {batch_acc:.4f} "
                  f"| RunningLoss: {running_loss_avg:.4f} "
                  f"| RunningAcc: {running_acc:.4f}")

    # End-of-epoch summary (adds TrainAcc)
    train_loss_epoch = total_loss
    train_acc_epoch = running_acc if running_seen > 0 else 0.0
    print(f"Epoch {epoch+1}, Training Loss (sum): {train_loss_epoch:.4f} | Training Acc: {train_acc_epoch:.4f}")

    # Validation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Epoch {epoch+1}, Validation Macro F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = model.state_dict()
        early_stop_counter = 0
        print("  Best model so far saved.")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("  Early stopping triggered.")
            break

# Save best model
torch.save(best_model_state, "best_mathbert_model.pt")

# Final evaluation
print("\n Classification Report:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

print("\n Confusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)

# Save confusion matrix as image
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
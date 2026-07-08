"""
stage2b_train_teacher.py (v2)
─────────────────────────────────────────────────────────────
Fine-tunes BERT-Base (teacher) on the de-masked, pii_types-free
ReCon data. Proper 3-way split: 72% train / 8% val / 20% test.
Reports confusion matrix after each validation epoch.
Saves model + teacher_train_probs.json for QAD distillation.
"""

import json
import time
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix
)
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MODEL_NAME   = "bert-base-uncased"
MAX_LENGTH   = 128
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
VAL_FRACTION = 0.1

TRAIN_FILE = "pipeline_data/train_clean.jsonl"
TEST_FILE  = "pipeline_data/test_clean.jsonl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReConDataset(Dataset):
    def __init__(self, items, tokenizer, max_length):
        self.texts  = [x["text"]  for x in items]
        self.labels = [x["label"] for x in items]
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True,
            max_length=self.max_length, padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def print_cm(cm):
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print(f"    Confusion Matrix:")
    print(f"    {'':>14} {'Pred Non-leak':>15} {'Pred Leak':>12}")
    print(f"    {'Actual Non-leak':>14} {tn:>15,} {fp:>12,}")
    print(f"    {'Actual Leak':>14} {fn:>15,} {tp:>12,}")
    print(f"    TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")


def evaluate(model, loader, name=""):
    model.eval()
    preds_all, labels_all, probs_all = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            preds_all.extend(preds.tolist())
            labels_all.extend(batch["label"].numpy().tolist())
            probs_all.extend(probs.tolist())
    elapsed = time.time() - t0

    acc  = accuracy_score(labels_all, preds_all)
    f1   = f1_score(labels_all, preds_all, zero_division=0)
    prec = precision_score(labels_all, preds_all, zero_division=0)
    rec  = recall_score(labels_all, preds_all, zero_division=0)
    cm   = confusion_matrix(labels_all, preds_all, labels=[0, 1])

    print(f"\n  [{name}]")
    print(f"    Accuracy:  {acc*100:.2f}%  |  Precision: {prec*100:.2f}%")
    print(f"    Recall:    {rec*100:.2f}%  |  F1:        {f1*100:.2f}%")
    print_cm(cm)
    print(f"    Latency: {(elapsed/len(labels_all))*1000:.3f} ms/sample")

    return {
        "accuracy": acc, "f1": f1, "precision": prec,
        "recall": rec, "confusion_matrix": cm.tolist(),
        "probs": probs_all, "labels": labels_all
    }


def main():
    print("=" * 65)
    print("STAGE 2b (v2): Fine-tuning BERT-Base teacher on CLEAN data")
    print("=" * 65)
    print(f"\nDevice: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    full_train = load_jsonl(TRAIN_FILE)
    test_items = load_jsonl(TEST_FILE)

    random.shuffle(full_train)
    val_n = int(len(full_train) * VAL_FRACTION)
    val_items   = full_train[:val_n]
    train_items = full_train[val_n:]

    # Print class distribution for each split (ma'am asked for this)
    print("\nClass distribution per split:")
    for name, items in [("TRAIN", train_items),
                        ("VAL",   val_items),
                        ("TEST",  test_items)]:
        n = len(items)
        leaks = sum(1 for x in items if x["label"] == 1)
        print(f"  {name}: {n:,} total | "
              f"PII={leaks:,} ({100*leaks/n:.2f}%) | "
              f"Non-PII={n-leaks:,} ({100*(n-leaks)/n:.2f}%)")

    train_ds = ReConDataset(train_items, tokenizer, MAX_LENGTH)
    val_ds   = ReConDataset(val_items,   tokenizer, MAX_LENGTH)
    test_ds  = ReConDataset(test_items,  tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(train_loader) * EPOCHS
    )

    print(f"\nFine-tuning {EPOCHS} epochs...")
    t_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for step, batch in enumerate(train_loader):
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            out  = model(input_ids=ids, attention_mask=mask, labels=labels)
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            ep_loss += out.loss.item()
            if step % 200 == 0:
                print(f"  Ep{epoch+1} step {step}/{len(train_loader)} "
                      f"loss={out.loss.item():.4f}")

        avg = ep_loss / len(train_loader)
        print(f"  Epoch {epoch+1} avg loss: {avg:.4f}")
        evaluate(model, val_loader, name=f"Validation epoch {epoch+1}")

    print(f"\nTraining time: {(time.time()-t_start)/60:.1f} min")

    print("\n" + "=" * 65)
    print("FINAL TEST SET RESULTS")
    print("=" * 65)
    test_res = evaluate(model, test_loader, name="Test")
    model_mb = sum(p.numel()*4 for p in model.parameters()) / (1024**2)
    print(f"\n  Model size (FP32): {model_mb:.2f} MB")

    # Save teacher model
    os.makedirs("teacher_model", exist_ok=True)
    model.save_pretrained("teacher_model")
    tokenizer.save_pretrained("teacher_model")
    print("\nSaved teacher model to teacher_model/")

    # Save teacher probabilities on TEST set
    with open("pipeline_data/teacher_test_probs.json", "w") as f:
        json.dump({"probs": test_res["probs"], "labels": test_res["labels"]}, f)

    # Also generate teacher probabilities on TRAIN set (needed for QAD)
    print("\nGenerating teacher probabilities on TRAIN set for QAD...")
    model.eval()
    train_probs_all, train_labels_all = [], []
    # Use the exact same train_items order (not shuffled again)
    train_ds_ordered = ReConDataset(train_items, tokenizer, MAX_LENGTH)
    train_ordered_loader = DataLoader(
        train_ds_ordered, batch_size=BATCH_SIZE, shuffle=False
    )
    with torch.no_grad():
        for batch in train_ordered_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()
            train_probs_all.extend(probs.tolist())
            train_labels_all.extend(batch["label"].numpy().tolist())

    with open("pipeline_data/teacher_train_probs.json", "w") as f:
        json.dump({"probs": train_probs_all,
                   "labels": train_labels_all,
                   "n_train": len(train_items)}, f)

    # Sanity check
    lp = np.mean([p[1] for p, l in zip(train_probs_all, train_labels_all) if l==1])
    np_ = np.mean([p[1] for p, l in zip(train_probs_all, train_labels_all) if l==0])
    print(f"  Sanity check — avg P(leak): leaks={lp:.4f}, non-leaks={np_:.4f}")
    print(f"  Saved {len(train_probs_all):,} train probs to teacher_train_probs.json")

    print("\nSTAGE 2b COMPLETE")


if __name__ == "__main__":
    main()
"""Fine-tune FP32 TinyBERT standalone — get accuracy, F1, latency."""
import json, time, os, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

random.seed(42); np.random.seed(42); torch.manual_seed(42)

MODEL_NAME   = "huawei-noah/TinyBERT_General_4L_312D"
MAX_LENGTH   = 128
BATCH_SIZE   = 32
EPOCHS       = 3
LR           = 2e-5
VAL_FRACTION = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReConDataset(Dataset):
    def __init__(self, items, tokenizer):
        self.texts  = [x["text"]  for x in items]
        self.labels = [x["label"] for x in items]
        self.tok    = tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], truncation=True, max_length=MAX_LENGTH,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}

def load_jsonl(path):
    with open(path) as f: return [json.loads(l) for l in f]

def print_cm(cm):
    tn,fp,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
    print(f"    TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")

def evaluate(model, loader, name=""):
    model.eval(); preds_all, labels_all = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            preds = torch.argmax(out.logits, dim=1).cpu().numpy()
            preds_all.extend(preds.tolist())
            labels_all.extend(batch["label"].numpy().tolist())
    elapsed = time.time()-t0
    acc  = accuracy_score(labels_all, preds_all)
    f1   = f1_score(labels_all, preds_all, zero_division=0)
    prec = precision_score(labels_all, preds_all, zero_division=0)
    rec  = recall_score(labels_all, preds_all, zero_division=0)
    cm   = confusion_matrix(labels_all, preds_all, labels=[0,1])
    print(f"\n  [{name}]")
    print(f"    Accuracy: {acc*100:.2f}%  Precision: {prec*100:.2f}%")
    print(f"    Recall:   {rec*100:.2f}%  F1:        {f1*100:.2f}%")
    print_cm(cm)
    print(f"    Latency: {(elapsed/len(labels_all))*1000:.3f} ms/sample")
    return {"accuracy":acc,"f1":f1,"precision":prec,"recall":rec}

def main():
    print("="*60)
    print("FP32 TinyBERT Standalone Fine-tuning")
    print("="*60)
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(device)

    full_train = load_jsonl("pipeline_data/train_clean.jsonl")
    test_items = load_jsonl("pipeline_data/test_clean.jsonl")
    random.shuffle(full_train)
    val_n = int(len(full_train)*VAL_FRACTION)
    val_items, train_items = full_train[:val_n], full_train[val_n:]

    print(f"\nSplit: Train={len(train_items):,} Val={len(val_n if False else val_items):,} Test={len(test_items):,}")
    for nm, items in [("TRAIN",train_items),("VAL",val_items),("TEST",test_items)]:
        n=len(items); l=sum(1 for x in items if x["label"]==1)
        print(f"  {nm}: {n:,} | PII={l:,}({100*l/n:.1f}%) NonPII={n-l:,}({100*(n-l)/n:.1f}%)")

    # Class weights
    n_leak = sum(1 for x in train_items if x["label"]==1)
    n_nonleak = len(train_items)-n_leak
    w = torch.tensor([len(train_items)/(2*n_nonleak),
                      len(train_items)/(2*n_leak)]).to(device)
    print(f"\nClass weights: NonLeak={w[0]:.3f} Leak={w[1]:.3f}")

    train_ds = ReConDataset(train_items, tokenizer)
    val_ds   = ReConDataset(val_items,   tokenizer)
    test_ds  = ReConDataset(test_items,  tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    import torch.nn as nn
    ce = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(train_loader)*EPOCHS)

    print(f"\nTraining {EPOCHS} epochs...")
    t_start = time.time()
    for epoch in range(EPOCHS):
        model.train(); ep_loss=0.0
        for step, batch in enumerate(train_loader):
            ids=batch["input_ids"].to(device)
            mask=batch["attention_mask"].to(device)
            labels=batch["label"].to(device)
            optimizer.zero_grad()
            out=model(input_ids=ids,attention_mask=mask)
            loss=ce(out.logits,labels)
            loss.backward(); optimizer.step(); scheduler.step()
            ep_loss+=loss.item()
            if step%200==0:
                print(f"  Ep{epoch+1} step{step}/{len(train_loader)} loss={loss.item():.4f}")
        print(f"  Epoch {epoch+1} avg loss: {ep_loss/len(train_loader):.4f}")
        evaluate(model, val_loader, name=f"Val epoch {epoch+1}")

    print(f"\nTraining time: {(time.time()-t_start)/60:.1f} min")
    print("\n"+"="*60)
    print("FINAL TEST RESULTS — FP32 TinyBERT")
    print("="*60)
    evaluate(model, test_loader, name="Test")
    mb = sum(p.numel()*4 for p in model.parameters())/(1024**2)
    print(f"\n  Model size (FP32): {mb:.2f} MB")

if __name__ == "__main__":
    main()

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate
import numpy as np

# ── Config ──────────────────────────────────────────────
MODEL_NAME   = "facebook/bart-large-cnn"
OUTPUT_DIR   = "model/results"
MAX_INPUT    = 512
MAX_TARGET   = 128
BATCH_SIZE   = 2
EPOCHS       = 1
LR           = 2e-5

# ── Load data ────────────────────────────────────────────
print("Loading dataset...")
dataset = load_dataset("ccdv/pubmed-summarization", "document")
train = dataset["train"].select(range(1000))
val   = dataset["validation"].select(range(200))

# ── Load tokenizer & model ───────────────────────────────
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print(f"Model loaded: {MODEL_NAME}")
print(f"Parameters  : {model.num_parameters():,}")

# ── Tokenize ─────────────────────────────────────────────
def tokenize(batch):
    inputs = tokenizer(
        batch["article"],
        max_length=MAX_INPUT,
        truncation=True,
        padding="max_length"
    )
    targets = tokenizer(
        batch["abstract"],
        max_length=MAX_TARGET,
        truncation=True,
        padding="max_length"
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

print("Tokenizing...")
train_tok = train.map(tokenize, batched=True, remove_columns=["article", "abstract"])
val_tok   = val.map(tokenize,   batched=True, remove_columns=["article", "abstract"])
print("Tokenization done.")

# ── ROUGE metric ─────────────────────────────────────────
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds   = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels  = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels,  skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {k: round(v, 4) for k, v in result.items()}

# ── Training args ─────────────────────────────────────────
args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    predict_with_generate=True,
    fp16=False,
    logging_steps=100,
    report_to="none"
)

# ── Trainer ───────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics
)

print("\nStarting training...")
trainer.train()

# ── Save ──────────────────────────────────────────────────
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
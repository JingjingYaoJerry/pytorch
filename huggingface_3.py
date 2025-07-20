"""
huggingface_3.py
by Jingjing YAO (Jerry)

An interactive command-line utility that wraps common workflows of the
🤗 Transformers library.  Users can:

* dataset download & tokenization (GLUE / custom),
* Trainer-based or manual training loops,
* optional Accelerate integration,
* evaluation with 🤗 Evaluate,
* save / push checkpoints.

The script is entirely self-contained; no external HTML, images, or other
assets are required.
"""

import os
import sys
import pathlib
from typing import Optional, Dict, Any

import torch

# ----------------------------- helpers ----------------------------- #
def dynamic_import(class_name: str):
    """Lightweight `from transformers import <class_name>`."""
    try:
        return getattr(__import__("transformers", fromlist=[class_name]),
                       class_name)
    except AttributeError as exc:
        raise ValueError(f"{class_name!r} not found in transformers") from exc


def yes(prompt: str) -> bool:
    return input(prompt).strip().upper() == "Y"


def save_to_dir(model, tokenizer, directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    model.save_pretrained(directory)
    if tokenizer is not None:
        tokenizer.save_pretrained(directory)
    print(f"✓ Saved model{' & tokenizer' if tokenizer else ''} to {directory}")


# ---------------------- data preparation --------------------------- #
def load_and_tokenize() -> Dict[str, Any]:
    """
    Download a dataset with 🤗 Datasets and tokenize it.
    Returns a dict:
      {
        "tokenized": DatasetDict,
        "data_collator": callable,
        "label_names": list[str],
        "num_labels": int,
      }
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding

    # --- dataset ----------------------------------------------------
    default_name = "glue"
    default_config = "mrpc"
    ds_name = input(f"Dataset name (default {default_name}): ").strip() or default_name
    ds_cfg = input(f"Dataset config (default {default_config}): ").strip() or default_config
    raw_datasets = load_dataset(ds_name, ds_cfg)
    print(raw_datasets)

    # --- tokenizer --------------------------------------------------
    ckpt = input("Base checkpoint for both tokenizer & model "
                 "(default bert-base-uncased): ").strip() or "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    # decide whether we have sentence pairs or single sentences
    sample = raw_datasets["train"][0]
    guess_cols = [c for c in sample if sample[c] and isinstance(sample[c], str)]
    if len(guess_cols) >= 2:
        sent1_col, sent2_col = guess_cols[:2]
    else:
        sent1_col, sent2_col = guess_cols[0], None
    print(f"◾ text columns detected: {sent1_col}"
          f"{' / ' + sent2_col if sent2_col else ''}")

    # tokenization function
    def tok_fn(ex):
        if sent2_col:
            return tokenizer(ex[sent1_col], ex[sent2_col],
                             truncation=True)
        return tokenizer(ex[sent1_col], truncation=True)

    tokenized = raw_datasets.map(tok_fn, batched=True, remove_columns=guess_cols + ["idx"]
                                 if "idx" in sample else guess_cols)
    data_collator = DataCollatorWithPadding(tokenizer)
    label_info = raw_datasets["train"].features["label"]
    label_names = label_info.names if hasattr(label_info, "names") else None
    num_labels = label_info.num_classes if hasattr(label_info, "num_classes") else 1

    return dict(
        tokenized=tokenized,
        data_collator=data_collator,
        label_names=label_names,
        num_labels=num_labels,
        checkpoint=ckpt,
        tokenizer=tokenizer,
    )


# ------------------------- Trainer branch -------------------------- #
def trainer_workflow(bundle: Dict[str, Any]):
    from transformers import (AutoModelForSequenceClassification, TrainingArguments,
                              Trainer)
    import evaluate
    import numpy as np

    ckpt = bundle["checkpoint"]
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, num_labels=bundle["num_labels"])

    # --- TrainingArguments -----------------------------------------
    out_dir = input("Output directory (default ./results): ").strip() or "./results"
    training_args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        report_to="none",     # tip: set to “wandb” to log curves
        push_to_hub=False,
    )

    # --- metrics ----------------------------------------------------
    metric = evaluate.load("glue", "mrpc") if bundle["label_names"] else None

    def compute_metrics(eval_pred):
        if metric is None:
            return {}
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=bundle["tokenized"]["train"],
        eval_dataset=bundle["tokenized"]["validation"],
        tokenizer=bundle["tokenizer"],
        data_collator=bundle["data_collator"],
        compute_metrics=compute_metrics,
    )

    print("🟢 Starting training with Trainer API …")
    trainer.train()

    print("📊 Final evaluation:")
    eval_res = trainer.evaluate()
    print(eval_res)

    if yes("Save this fine-tuned checkpoint? (Y/N) "):
        save_to_dir(model, bundle["tokenizer"],
                    input("Dir (default ./finetuned): ").strip() or "./finetuned")

    print("💡  Tip: set TrainingArguments.report_to='wandb' "
          "and watch learning curves in real time (see 3_4.txt).")


# ---------------------- manual loop branch ------------------------ #
def custom_loop(bundle: Dict[str, Any]):
    from transformers import (AutoModelForSequenceClassification, get_scheduler)
    from torch.utils.data import DataLoader
    import evaluate
    from tqdm.auto import tqdm

    use_accelerate = yes("Use 🤗 Accelerate for distributed / mixed-precision? (Y/N) ")
    if use_accelerate:
        try:
            from accelerate import Accelerator
        except ModuleNotFoundError:
            print("Accelerate not installed; falling back to pure PyTorch.")
            use_accelerate = False

    # dataloaders ----------------------------------------------------
    train_dl = DataLoader(bundle["tokenized"]["train"], shuffle=True,
                          batch_size=8, collate_fn=bundle["data_collator"])
    eval_dl = DataLoader(bundle["tokenized"]["validation"], batch_size=8,
                         collate_fn=bundle["data_collator"])

    model = dynamic_import("AutoModelForSequenceClassification").from_pretrained(
        bundle["checkpoint"], num_labels=bundle["num_labels"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    if use_accelerate:
        accelerator = Accelerator()
        model, optimizer, train_dl, eval_dl = accelerator.prepare(
            model, optimizer, train_dl, eval_dl
        )
        device = accelerator.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    num_epochs = int(input("Epochs (default 3): ").strip() or 3)
    num_steps = num_epochs * len(train_dl)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=num_steps)

    metric = evaluate.load("glue", "mrpc") if bundle["label_names"] else None

    progress = tqdm(range(num_steps))
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if use_accelerate:
                accelerator.backward(loss)
            else:
                loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress.update(1)

        # --- simple eval each epoch --------------------------------
        model.eval()
        metric.reset() if metric else None
        with torch.no_grad():
            for batch in eval_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                if metric:
                    preds = torch.argmax(outputs.logits, dim=-1)
                    metric.add_batch(predictions=preds,
                                     references=batch["labels"])
        if metric:
            print(f"\n🔎 Epoch {epoch+1}: {metric.compute()}")
        model.train()

    if yes("Save final checkpoint? (Y/N) "):
        path = input("Directory (default ./finetuned): ").strip() or "./finetuned"
        if use_accelerate:
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            save_to_dir(unwrapped, bundle["tokenizer"], path)
        else:
            save_to_dir(model, bundle["tokenizer"], path)

    print("🏁 Training finished.  "
          "Plot the logged losses in W&B or another tool to inspect "
          "learning curves (cf. 3_4.txt).")


# ------------------------------- main ------------------------------ #
def main():
    print("=== Fine-tuning playground (Ch. 3) ===")
    bundle = load_and_tokenize()

    choice = input(
        "Choose workflow: (trainer/custom) "
    ).strip().lower() or "trainer"

    if choice == "trainer":
        trainer_workflow(bundle)
    else:
        custom_loop(bundle)


# ------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
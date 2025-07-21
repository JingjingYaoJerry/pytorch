"""
Please install the `datasets` library beforehand to run this script:
`pip install -U datasets`

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

    # ==================== 1. Load Dataset ====================== #
    print("As demonstrated in the tutorial, a MRPC DatasetDict from the GLUE benchmark can be downloaded with:")
    print("raw_datasets = load_dataset('glue', 'mrpc')")
    print("But upon checking, it appears to be a legacy method based on its latest documentation and `load_dataset`'s documentation:")
    print("https://huggingface.co/datasets/nyu-mll/glue?library=datasets")
    print("https://huggingface.co/docs/datasets/v1.13.2/package_reference/loading_methods.html#datasets.load_dataset")
    input("Press enter to load the dataset using `ds = load_dataset('nyu-mll/glue', 'mrpc')` ...")
    raw_datasets = load_dataset("nyu-mll/glue", "mrpc")
    print(f"By default the dataset is going to be downloaded to {pathlib.Path.home() / '.cache' / 'huggingface' / 'datasets'}")
    print("To change the cache directory, set the environment variable `HF_HOME`.")
    input("Press Enter to continue...")

    # ==================== 2. Split Dataset ====================== #
    print(f"Returned dataset (DatasetDict): \n{raw_datasets}")
    print("As demonstrated, the DatasetDict has already been split into train / validation / test Datasets.")
    print("By indexing, we can split it into train / validation / test sets (e.g., raw_datasets['train']): ")
    raw_train_dataset, raw_validation_dataset, raw_test_dataset = raw_datasets["train"], raw_datasets["validation"], raw_datasets["test"]
    print(f"raw_datasets['train']: {raw_train_dataset}")
    print("Where each datapoint can be accessed by indexing as well (e.g., raw_datasets['train'][0]): ")
    print(f"The train set's first datapoint: {raw_train_dataset[0]}")
    input("Press Enter to continue...")
    if yes("Check the dataset's features? (Y/N) "):
        print("Using `raw_train_dataset.features` to inspect the features of the dataset...")
        print(raw_train_dataset.features)
        print("Where the `ClassLabel` type is the mapping of the label names to integers (0 == not_equivalent).")
        input("Press Enter to continue...")
    
    #============ Dataset Check ============#
    if yes("Have you noticed that the idx of the validation set's datapoints are not in order? (Y/N) "):
        print("It appears that the datapoints in the validation set are taken randomly from the train set...")
        print("Where the test set is on its own, aren't they?")
    else:
        print("Lookup the dataset on Hugging Face Datasets to check its information.")
        print('e.g., in this case lookup the `mrpc` "subset" under the `nyu-mll/glue` "dataset"')
    input("Press Enter to continue...")

    # ==================== 3. Tokenization ====================== #
    ckpt = input("Checkpoint for both tokenizer & model "
                 "(default bert-base-uncased): ").strip() or "bert-base-uncased"
    print("Using AutoTokenizer here for demonstration purposes")
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    print("Being a paraphrase corpus, `mrpc` needs to be handled as a sentence pair classification task...")
    print("Where it appears that not only Bert's tokenizer can handle more than one sentence (e.g., GPT-2).")
    print("In BERT's case, the returned input ids are merged into a single sequence...")
    print("Where the token_type_ids are used to differentiate the two sentences specifically for BERT's 'next sentence prediction' tasks.")
    print("E.g., when two sentences `This is the first sentence.` and `This is the second sentence.` are tokenized with" \
    "`tokenizer('This is the first sentence.', 'This is the second sentence.')`, below is the output:")
    print(tokenizer("This is the first sentence.", "This is the second sentence."))
    input("Press Enter to continue...")

    #============ Token Type IDs Check ============#
    print("See the `token_type_ids` are formed with 8 zeros followed by 8 ones?")
    print("Recall that a BERT tokenizer uses the special tokens [CLS] and [SEP]?")
    input("Press Enter to continue...")

    #============ Special Tokens Check ============#
    if yes("Do you know the difference between `tokenizer.decode` & `tokenizer.convert_ids_to_tokens`? (Y/N) "):
        pass
    else:
        print("The `tokenizer.decode` method decodes the input ids directly into a string, "
              "while `tokenizer.convert_ids_to_tokens` converts the input ids (in a list) into tokens (in a list).")
        print("E.g., `tokenizer.decode([101, 2023, 2003, 1996, 2468, 102])` returns "
              "'[CLS] this is the first sentence. [SEP]', while "
              "`tokenizer.convert_ids_to_tokens([101, 2023, 2003, 1996, 2468, 102])` returns "
              "`['[CLS]', 'this', 'is', 'the', 'first', 'sentence.', '[SEP]']`.")
        input("Press Enter to continue...")

    print("For simplicity, `tokenizer(raw_train_dataset['sentence1'], raw_train_dataset['sentence2'], " \
    "padding=True, truncation=True)` can be used directly to tokenize all datapoints.")
    print("But considering its returned 'lists of lists' and RAM usage, Dataset.map() is suggested.")

    

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
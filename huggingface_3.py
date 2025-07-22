"""
Please install the `datasets` and `evaluate` libraries beforehand to run this script:
`pip install -U datasets evaluate`

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
from colorama import Fore, Style
from sympy import evaluate

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


# ---------------------- Trainer Branch --------------------------- #
def Trainer():
    """
    Fine-tune a model with `transformers.Trainer`.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

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
    print("But considering its returned 'lists of lists' and RAM usage, Dataset.map() is suggested...")
    print("Which can apply the tokenization function (or more) to each datapoint in the dataset, just like the `map` function in Python~")
    print("i.e., ")
    print(Fore.BLUE)
    print("def tokenize_function(example):")
    print("\treturn tokenizer(example['sentence1'], example['sentence2'], truncation=True)")
    print("tokenized_datasets = raw_datasets.map(tokenize_function, batched=True) # batched=True enables batch processing of the function")
    print(Style.RESET_ALL)
    
    #============ Padding Check ============#
    if yes("Does the setting of the `padding` argument here make difference? (Y/N) "):
        print("Yes, it does; do recall both the efficiency of padding in different places and the model's requirement of rectangular tensors.") 
    else:
        print("Not really, the `padding` here would pad all samples to one maximum length...")
        print("Recall that one batch is processed at a time, and hence the most efficient way is to pad every sample in the " \
              "batch to the maximum length of that batch, instead of padding all samples to the maximum length of the entire " \
              "dataset to satisfy the rectangular tensor requirement from the model.")
    input("Press Enter to continue...")

    # Tokenization Function Mapping
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    tokenized = raw_datasets.map(tokenize_function, batched=True)
    print("This way all three splits are tokenized and having input_ids, attention_mask, and token_type_ids.")
    print("i.e., ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask']")
    input("Press Enter to continue...")

    # ==================== 4. Define Collator ====================== #
    if yes("Do you know about the collate function? (Y/N) "):
        if yes("Fantastic! You know how it's used, right?"):
            print("Great! Now we will follow the logic to create a collator for dynamic padding with the `DataCollatorWithPadding`.")
        else:
            print("`collate_fn` is a parameter of the pytorch DataLoader responsible for batching the samples.")
    else:
        print("The `collate_fn` is a function that takes a list of samples and returns a batch.")
        print("It is used to pad the samples to the same length, so they can be processed in a batch.")
        print("In this case, we will use the `DataCollatorWithPadding` from the Transformers library to handle dynamic padding.")
        print("It will pad the samples to the maximum length of the batch, instead of the maximum length of the entire dataset.")
    input("Press Enter to continue...")
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print("`DataCollatorWithPadding(tokenizer=tokenizer)` is used to return the collator.")

    # ==================== 5. Construct Trainer ====================== #
    from transformers import TrainingArguments
    print("Four steps to construct a Trainer:")
    print(Fore.GREEN + "1. Define the training hyperparameters with `TrainingArguments`:" + Style.RESET_ALL)
    if yes("Do you know what's the must-have argument for `TrainingArguments`? (Y/N) "):
        print("Do recall that `output_dir` is the directory where the model predictions and checkpoints will be saved.")
    else:
        print("The must-have argument for `TrainingArguments` is `output_dir`.")
    print("It is the directory where the model predictions and checkpoints will be saved.")
    input("Press Enter to continue...")
    print("The default output_dir `trainer_output` is used here for demonstration purposes.")

    #============ Training Arguments Demo ============#
    print("Check all available parameters for `TrainingArguments` in its documentation:")
    print("https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments")
    print("One parameter that was demonstrated is `eval_strategy`; by setting it to `epoch`, an evaluation would occur at the end of each epoch rather than after some steps.")
    print("Another parameter is `fp16`, which is used to enable mixed precision training with 16-bit floating-point numbers for faster training and reduced memory usage.")
    print("Another parameter is `gradient_accumulation_steps`, which is used to accumulate gradients over the specified # of batches before performing backpropagation.")
    print("Calling `TrainingArguments('trainer_output', eval_strategy='epoch', fp16=True, gradient_accumulation_steps=4)` for demonstration...")
    training_args = TrainingArguments("trainer_output", eval_strategy="epoch", fp16=True, gradient_accumulation_steps=4)
    input("Press Enter to continue...")
    
    print(Fore.GREEN + "2. Define the model:" + Style.RESET_ALL)
    from transformers import AutoModelForSequenceClassification
    print(Fore.BLUE + "Calling `model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)` for demonstration..." + Style.RESET_ALL)
    print(Fore.GREEN + "A warning may appear here just to indicate that the model needs to be fine-tuned for the specified task." + Style.RESET_ALL)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)

    print(Fore.GREEN + "3. Deploy the metrics:" + Style.RESET_ALL)
    print("Metrics are used to evaluate the model's performance during training and evaluation.")
    print("To get metrics other than simple losses, `compute_metrics()` is suggested to be defined...")
    print("Where it is to be defined as below:")
    print(Fore.BLUE)
    print("def compute_metrics(eval_preds):")
    print("\tmetric = evaluate.load('glue', 'mrpc') # or any other metric")
    print("\tlogits, labels = eval_preds")
    print("\tpredictions = np.argmax(logits, axis=-1) # or any other way to get predictions")
    print("\treturn metric.compute(predictions=predictions, references=labels)")
    print(Style.RESET_ALL)

    #============ evaluate.load Check ============#
    if yes("Do you know why it is 'glue' here rather than 'nyu-mll/glue'? (Y/N) "):
        pass
    else:
        print("According to `evaluate.load`'s documentation: ")
        print("https://huggingface.co/docs/evaluate/v0.1.2/en/package_reference/loading_methods#evaluate.load.path")
        print("The first argument is either a local path to a metric script or a evaluation identifier on the `evaluate` repo...")
        print("Where all available metrics can be found at https://github.com/huggingface/evaluate/tree/main/metrics")
        print("i.e., `glue` is one of the available identifiers, where `mrpc` is the `config_name` for the subset.")
    input("Press Enter to continue...")

    #============ Logits Check ============#
    if yes("Can you recall what `np.argmax(logits, axis=-1)` does? (Y/N) "):
        pass
    else:
        print("Do recall that a Transformers model outputs logits, which are the raw unnormalized predictions that need post-processing.")
        print("The shape of the logits is going to be something like (batch_size, num_labels), right?")
        print("Here what we are interested in is the predicted label for each sample, where we can then use them to compare with the true labels, right?")
        print("So by calling `np.argmax(logits, axis=-1)`, we are getting the index of the maximum value in each row (i.e., among the logits) of the sample array, right?")
    print("Also, do recall that by calling `trainer.predict()`, predictions, label_ids, and metrics (with just losses) are returned by the Trainer.")
    input("Press Enter to continue...")
    from evaluate import load
    def compute_metrics(eval_preds):
        metric = evaluate.load("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    print("By defining the `compute_metrics` function, the trainer can report the validation loss and metrics at the end of each epoch on top of the training loss.")
    print(Fore.GREEN + "It'd be the post-processing function for the predictions returned by the Trainer." + Style.RESET_ALL)
    input("Press Enter to continue...")

    print(Fore.GREEN + "4. Define the Trainer:" + Style.RESET_ALL)
    print("Do recall that the Trainer is a high-level API for training and evaluating models.")
    if yes("Do you remember which other arguments are required for the Trainer? (Y/N) "):
        pass
    else:
        print("Some other required arguments for the Trainer are:")
        print(Fore.BLUE + "`train_dataset` & `eval_dataset` & `data_collator` & `processing_class`" + Style.RESET_ALL)
    input("Press Enter to continue...")

    #============ eval_dataset Check ============#
    if yes("Is the missing of `eval_dataset` going to cause an error? (Y/N) "):
        print("Not really, do recall that what the evaluation is for during training, and which component is used for the updating of the model's weights!!!")
    else:
        print("That's right, though no metrics will be reported during training, the model's weights will still be updated and the training carries on.")
    input("Press Enter to continue...")

    print(Fore.BLUE + "Calling `trainer = Trainer(model=model, args=training_args, train_dataset=tokenized['train'], eval_dataset=tokenized['validation'], "
    "data_collator=data_collator, processing_class=tokenizer, compute_metrics=compute_metrics)` for demonstration..." + Style.RESET_ALL)
    if yes("Do you remember what the `processing_class` is? (Y/N) "):
        if yes("Great! Then you know how it's affecting the passing of the `data_collator` argument, right? (Y/N) "):
            pass
        else:
            print("Once a tokenizer is passed to the `processing_class`, the `data_collator` argument is automatically set to `DataCollatorWithPadding(tokenizer)`.")
            print("Hence, the data_collator we've defined earlier is no longer needed to be passed explicitly.")
            print("i.e., `data_collator=data_collator` is not really needed in the Trainer.")
    else:
        print("The `processing_class` is the class that processes (i.e., dynamically pads) the data before passing it to the model.")
        print("Hence, the `data_collator` argument is automatically set to `DataCollatorWithPadding(tokenizer)`.")
        print("i.e., `data_collator=data_collator` is not really needed in the Trainer.")
    input("Press Enter to continue...")
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        data_collator=data_collator,
        processing_class=tokenizer, 
        compute_metrics=compute_metrics
    )

    # ==================== 6. Fine-Tune ====================== #
    print("Now we can train the model with the Trainer's `train()` method.")
    print(Fore.BLUE + "Calling `trainer.train()` for demonstration..." + Style.RESET_ALL)
    trainer.train()
    predictions = trainer.predict(tokenized['test'])
    return predictions


    print("💡  Tip: set TrainingArguments.report_to='wandb' "
          "and watch learning curves in real time (see 3_4.txt).")

# ---------------------- PyTorch Branch ------------------------ #
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
"""
Please install the `datasets` and `evaluate` Hugging Face libraries beforehand to run this script:
`pip install -U datasets evaluate`
It'd be helpful to have the third-party `wandb` library installed as well for logging and visualization:
`pip install -U wandb`

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


def print_code(code: str) -> None:
    """Print code with syntax highlighting."""
    print(Fore.BLUE + code + Style.RESET_ALL)


def highlight(text: str) -> str:
    """Highlight text with color."""
    return Fore.GREEN + text + Style.RESET_ALL


# ---------------------- Datasets --------------------------- #
def load():
    """
    Load a dataset

    Returns
    -------
    Dataset or DatasetDict or IterableDataset or IterableDatasetDict
        The dataset loaded with `datasets.load_dataset()`.
    """
    from datasets import load_dataset
    print("As demonstrated in the tutorial, a MRPC DatasetDict from the GLUE benchmark can be downloaded with:")
    print_code("raw_datasets = load_dataset('glue', 'mrpc')")
    highlight("Where the first parameter `path`'s argument can be a dataset on the HF Hub.")
    print("But upon checking, it appears to be a legacy method based on its latest documentation and `load_dataset`'s documentation:")
    print("https://huggingface.co/datasets/nyu-mll/glue?library=datasets")
    print("https://huggingface.co/docs/datasets/v1.13.2/package_reference/loading_methods.html#datasets.load_dataset")
    print("OR...")
    highlight("In addition to arguments of local datasets (e.g., './dataset/squad' or './dataset/squad/squad.py')...")
    highlight("According to the same doc., if an argument of a 'generic dataset' is passed (e.g., csv, json, etc.), a generic dataset builder will be returned by `load_dataset`.")
    input("Press enter to load the dataset using" + Fore.BLUE + "ds = load_dataset('nyu-mll/glue', 'mrpc')" + Style.RESET_ALL + " ...")
    raw_datasets = load_dataset("nyu-mll/glue", "mrpc")
    highlight(f"By default the dataset is going to be downloaded to {pathlib.Path.home() / '.cache' / 'huggingface' / 'datasets'}")
    highlight("To change the cache directory, set the environment variable `HF_HOME`.")

    return raw_datasets
    
# ---------------------- Trainer Branch --------------------------- #
def Trainer():
    """
    Fine-tune a model with `transformers.Trainer`.
    """
    from transformers import AutoTokenizer

    # ==================== 1. Load Dataset ====================== #
    raw_datasets = load()
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

    #============ wandb Demo ============#
    print("According to the tutorial, it's displayed that the `wandb` module, the `report_to` & `logging_steps` arguments in the Trainer API can be used together for logging and visualization.")
    if yes("Would you like to give it a try? (Y/N) "):
        print("To use `wandb`, firstly we need to initialize it with `wandb.init()`.")
        print(Fore.BLUE + "Executing `import wandb`..." + Style.RESET_ALL)
        import wandb
        print(Fore.BLUE + "Executing `wandb.init(project='transformer-fine-tuning', name='bert-mrpc-analysis')`..." + Style.RESET_ALL)
        print("Where referring to its documentation `https://docs.wandb.ai/ref/python/sdk/functions/init/`, `project` is the name of the project under which this run will be logged, " \
        "and `name` is the display name for this run which appears in the UI.")
        wandb.init(project="transformer-fine-tuning", name="bert-mrpc-analysis")
        print(Fore.BLUE + "Executing `training_args = TrainingArguments('trainer_output', eval_strategy='epoch', fp16=True, gradient_accumulation_steps=4, logging_steps=10, report_to='wandb')`..." + Style.RESET_ALL)
        training_args = TrainingArguments("trainer_output", eval_strategy="epoch", fp16=True, gradient_accumulation_steps=4, logging_steps=10, report_to="wandb")
    input("Let's wait and see how is it going to be displayed in the WandB UI... Press Enter to continue...")

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
def custom_torch():
    """
    Fine-tune a model with a custom PyTorch training loop.
    """
    #============ Trainer Branch Check ============#
    print("To fine-tune a model with PyTorch instead of `transformers.Trainer`, the Hugging Face dataset needs to be preprocessed first.")
    print("Follow the same first few steps as in the Trainer branch, including loading the dataset and tokenizing it.")
    print("Executing the following codes...")
    print(Fore.BLUE)
    print("from datasets import load_dataset")
    print("from transformers import AutoTokenizer, DataCollatorWithPadding")
    print("raw_datasets = load_dataset('nyu-mll/glue', 'mrpc')")
    print("ckpt = input('Checkpoint for both tokenizer & model (default bert-base-uncased): ').strip() or 'bert-base-uncased'")
    print("tokenizer = AutoTokenizer.from_pretrained(ckpt)")
    print("def tokenize_function(example):")
    print("\treturn tokenizer(example['sentence1'], example['sentence2'], truncation=True)")
    print("tokenized = raw_datasets.map(tokenize_function, batched=True)")
    print("data_collator = DataCollatorWithPadding(tokenizer=tokenizer)")
    print(Style.RESET_ALL)
    print("Do recall concepts behind `load_dataset`, 'Dynamic Padding', `batched=True`, and `DataCollatorWithPadding`!")
    # ==================== 1. Dataset Loading & Tokenization ====================== #
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding
    raw_datasets = load_dataset("nyu-mll/glue", "mrpc")
    ckpt = input("Checkpoint for both tokenizer & model "
                 "(default bert-base-uncased): ").strip() or "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    tokenized = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    input("Press Enter to continue...")

    # ==================== 2. Preprocessing ====================== #
    print("To compile PyTorch's training loop, the following steps are required (which can all be done with the DatasetDict's methods):")
    print(Fore.GREEN + "1. Remove the unnecessary 'columns' from the DatasetDict:" + Style.RESET_ALL)
    print("Do recall that the DatasetDict contains 'sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', and 'attention_mask' columns after tokenization!")
    print(Fore.BLUE + "Executing `tokenized = tokenized.remove_columns(['sentence1', 'sentence2', 'idx'])`..." + Style.RESET_ALL)
    tokenized = tokenized.remove_columns(['sentence1', 'sentence2', 'idx'])
    print(Fore.GREEN + "2. Rename the column `label` to `labels` as expected by PyTorch:" + Style.RESET_ALL)
    print(Fore.BLUE + "Executing `tokenized = tokenized.rename_column('label', 'labels')`..." + Style.RESET_ALL)
    tokenized = tokenized.rename_column("label", "labels")
    print(Fore.GREEN + "3. Set the format of the lists within the DatasetDict to PyTorch tensors:" + Style.RESET_ALL)
    print(Fore.BLUE + "Executing `tokenized.set_format('torch')`..." + Style.RESET_ALL)
    tokenized.set_format("torch")
    print(f"Now the DatasetDict has the following columns: {tokenized.column_names}")
    print(f"The {type(tokenized)} has become:")
    print(tokenized)
    input("Press Enter to continue...")

    # ==================== 3. DataLoader Creation ====================== #
    print("Now we can create the DataLoader.")
    print(Fore.BLUE + "Executing `from torch.utils.data import DataLoader`..." + Style.RESET_ALL)
    from torch.utils.data import DataLoader
    print(Fore.BLUE + "Executing `train_dataloader = DataLoader(tokenized['train'], shuffle=True, batch_size=8, collate_fn=data_collator)`..." + Style.RESET_ALL)
    train_dataloader = DataLoader(tokenized["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
    print(Fore.BLUE + "Executing `eval_dataloader = DataLoader(tokenized['validation'], batch_size=8, collate_fn=data_collator)`..." + Style.RESET_ALL)
    eval_dataloader = DataLoader(tokenized["validation"], batch_size=8, collate_fn=data_collator)

    #============ Batch Inspection Demo ============#
    if yes("Do you want to inspect the first batch of the train dataloader? (Y/N) "):
        print("Below codes can be used to inspect the first batch of a dataloader (i.e., whether there's any mistake in the process):")
        print(Fore.BLUE + "for batch in train_dataloader:")
        print("\tprint({k: v.shape for k, v in batch.items()})")
        print("\tbreak")
        print(Style.RESET_ALL)
        for batch in train_dataloader:
            print({k: v.shape for k, v in batch.items()})
            break

    # ==================== 4. Model Initialization ====================== #
    from transformers import AutoModelForSequenceClassification
    print("Model's initialization is the same as in the Trainer branch.")
    print(Fore.BLUE + "Executing `model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)`..." + Style.RESET_ALL)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)

    #============ Model Inspection Demo ============#
    if yes("Would you like to check whether the model is correctly initialized? (Y/N) "):
        print("Below codes can be used to inspect it (with the first batch)")
        print(Fore.BLUE + "for batch in train_dataloader:")
        print("\toutputs = model(**batch)")
        print("\tprint(outputs.loss, outputs.logits.shape)")
        print("\tbreak")
        print(Style.RESET_ALL)
        for batch in train_dataloader:
            outputs = model(**batch)
            print(outputs.loss, outputs.logits.shape)
            break
    input("Press Enter to continue...")

    # ==================== 5. Hyperparameter Setup ====================== #
    print("To achieve the same training efficiency as in the Trainer branch, we need to manually set up the optimizer and learning rate scheduler.")
    print(Fore.GREEN + "1. Define the optimizer:" + Style.RESET_ALL)
    print("Since the Trainer API utilizes AdamW optimizer with lr=5e-5, we will use it here as well.")
    print(Fore.BLUE + "Executing `from torch.optim import AdamW`..." + Style.RESET_ALL)
    print(Fore.BLUE + "Executing `optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)`..." + Style.RESET_ALL)
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=5e-5)
    input("Press Enter to continue...")

    #============ Optimizer Optimization Demo ============#
    if yes("Would you like to improve the training efficiency even further? (Y/N) "):
        print("Applying `weight_decay=0.01` to reduce overfitting and improve generalization.")
        lr = float(input("Probably a different learning rate (default 5e-5): ").strip() or 5e-5)
        print(Fore.BLUE + "Executing `optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)`..." + Style.RESET_ALL)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    input("Press Enter to continue...")

    print(Fore.GREEN + "2. Define the # of epochs:" + Style.RESET_ALL)
    print("Since the Trainer API uses 3 epochs by default, we will use it here as well.")
    print(Fore.BLUE + "Executing `num_epochs = 3`..." + Style.RESET_ALL)
    num_epochs = 3
    input("Press Enter to continue...")

    print(Fore.GREEN + "3. Define the learning rate scheduler:" + Style.RESET_ALL)
    print("The Trainer API uses a linear scheduler with a decay from the maximum value (5e-5) to 0...")
    print("Where the number of training steps is the product of the # of epochs and the # of batches.")
    print("To implement it, below codes can be used:")
    print(Fore.BLUE + "from transformers import get_scheduler" + Style.RESET_ALL)
    from transformers import get_scheduler
    print(Fore.BLUE + "num_training_steps = num_epochs * len(train_dataloader)" + Style.RESET_ALL)
    num_training_steps = num_epochs * len(train_dataloader)
    print(f"The number of training steps is {num_training_steps}.")
    print(Fore.BLUE + "lr_scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)" + Style.RESET_ALL)
    print("Executing...")
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    input("Press Enter to continue...")

    # ==================== 6. CPU to GPU ====================== #
    print("Like for every PyTorch 'model', move the model to the GPU if available.")
    print(Fore.BLUE + "Executing `device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')`..." + Style.RESET_ALL)
    print(Fore.BLUE + "Executing `model.to(device)`..." + Style.RESET_ALL)
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
    model.to(device)
    input("Press Enter to continue...")

    # ==================== 7. Metrics Deployment ====================== #
    print("Like in the Trainer branch, we can deploy the metrics with `evaluate.load`.")
    print(Fore.BLUE + "Executing `import evaluate`..." + Style.RESET_ALL)
    import evaluate
    print(Fore.BLUE + "Executing `metric = evaluate.load('glue', 'mrpc')`..." + Style.RESET_ALL)
    metric = evaluate.load("glue", "mrpc")
    print("Do recall that the `metric` is used to compute the metrics during evaluation, "
          "and the `compute_metrics` function is used to process the predictions and labels.")
    print("Also, do recall where the available metrics can be found.")
    input("Press Enter to continue...")

    # ==================== 8. Training ====================== #
    print("Now we can start the training loop.")
    print(Fore.BLUE + "Executing `from tqdm.auto import tqdm`..." + Style.RESET_ALL)
    from tqdm.auto import tqdm
    print(Fore.BLUE + "Executing `progress_bar = tqdm(range(num_training_steps))`..." + Style.RESET_ALL)
    progress_bar = tqdm(range(num_training_steps))
    print(Fore.BLUE + "Executing `model.train()`..." + Style.RESET_ALL)
    model.train() # set the model to training mode
    if yes("Do you recall what `model.train()` is for? (Y/N) "):
        pass
    else:
        print(Fore.GREEN + "`model.train()` is used to set the model to training mode." + Style.RESET_ALL)
    input("Press Enter to continue...")
    print(Fore.BLUE + "Executing for epoch in range(num_epochs):" + Style.RESET_ALL)
    print(Fore.BLUE + "\tfor batch in train_dataloader:" + Style.RESET_ALL)
    print(Fore.BLUE + "\t\tbatch = {k: v.to(device) for k, v in batch.items()} # move the batch to GPU" + Style.RESET_ALL)
    print(Fore.BLUE + "\t\toutputs = model(**batch)" + Style.RESET_ALL)
    print(Fore.BLUE + "\t\tloss = outputs.loss # without the logits" + Style.RESET_ALL)
    print(Fore.BLUE + "\t\tloss.backward() # compute gradients for those parameters that have `requires_grad=True`" + Style.RESET_ALL)
    print(Fore.BLUE + "\t\toptimizer.step() # update the model parameters" + Style.RESET_ALL)
    print(Fore.BLUE + "\t\tlr_scheduler.step() # update the learning rate" + Style.RESET_ALL)
    print(Fore.BLUE + "\t\toptimizer.zero_grad() # reset the gradients to zero to prevent gradient accumulation " + Style.RESET_ALL)
    print(Fore.BLUE + "\t\tprogress_bar.update(1)" + Style.RESET_ALL)
    if yes("Are you familiar with each line of the above codes? (Y/N) "):
        print("Great!")
    else:
        print("Take a look at the comments in the above codes to understand each line!")
    #============ Training Optimization Demo ============#
    if yes("Would you like to see some optimization tips for the training loop? (Y/N) "):
        print("1. Gradient Clipping: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`.")
        print("2. Use `torch.cuda.amp.autocast()` and `GradScaler` for faster training.")
        print("3. Gradient Accumulation: Accumulate gradients over multiple batches to simulate larger batch sizes.")
        print("4. Checkpointing: Save model checkpoints periodically to resume training if interrupted.")
    input("Press Enter to continue...")
    print("Since the evaluation is done at the end of each epoch...")
    print(Fore.BLUE + "Executing `model.eval() # set the model to evaluation mode`..." + Style.RESET_ALL)
    print(Fore.BLUE + "# Disable gradient computations to save memory and computation" + Style.RESET_ALL)
    print(Fore.BLUE + "Executing `with torch.no_grad():`..." + Style.RESET_ALL)
    print(Fore.BLUE + "\tfor batch in eval_dataloader:" + Style.RESET_ALL)
    print(Fore.BLUE + "\t\tbatch = {k: v.to(device) for k, v in batch.items()}" + Style.RESET_ALL)
    print(Fore.BLUE + "\t\toutputs = model(**batch)" + Style.RESET_ALL)
    print(Fore.BLUE + "\t\tlogits = outputs.logits # get the logits for comparing with the labels" + Style.RESET_ALL)
    print(Fore.BLUE + "\t\tpreds = torch.argmax(logits, dim=-1) # get the index of the maximum value among each sample's logits" + Style.RESET_ALL)
    print(Fore.BLUE + "\t\t# Use `add_batch` to accumulate the batch-level metrics for the overall metrics at the end of the epoch" + Style.RESET_ALL)
    print(Fore.BLUE + "\t\tmetric.add_batch(predictions=preds, references=batch['labels'])" + Style.RESET_ALL)
    print(Fore.BLUE + "\tmetric.compute() # Compute the metrics for the current epoch" + Style.RESET_ALL)
    print(Fore.BLUE + "Executing `model.train()`..." + Style.RESET_ALL)
    if yes("Are you familiar with each line of the above codes? (Y/N) "):
        print("Fantastic!")
    else:
        print("Take a look at the comments in the above codes to understand each line!")
    input("Press Enter to execute the training loop at once...")
    # One epoch at a time
    for epoch in range(num_epochs):
        # One batch at a time
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()} # move the batch to GPU
            outputs = model(**batch)
            loss = outputs.loss # without the logits
            loss.backward() # compute gradients for those parameters that have `requires_grad=True`

            optimizer.step() # update the model parameters
            lr_scheduler.step() # update the learning rate
            optimizer.zero_grad() # reset the gradients to zero to prevent gradient accumulation
            progress_bar.update(1)

        # ==================== 9. Evaluation ====================== #
        model.eval() # set the model to evaluation mode
        # Disable gradient computations to save memory and computation
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                
                logits = outputs.logits # get the logits for comparing with the labels
                preds = torch.argmax(logits, dim=-1) # get the index of the maximum value among each sample's logits
                # Use `add_batch` to accumulate the batch-level metrics for the overall metrics at the end of the epoch
                metric.add_batch(predictions=preds, references=batch["labels"])
            metric.compute() # Compute the metrics for the current epoch
        model.train()

    print("Congratulations! The model has been fine-tuned with PyTorch!")
    print("Hooray! 🎉")

    if yes("Would you like some desserts (^_^)? (Y/N) "):
        print("Great! Here's some extra on training with multiple GPUs or TPUs (i.e., distributed training):")
        print("By utilizing the `accellerate` library, distributed training can be achieved with changes in 3 places.")
        print("An example is presented below:")
        print(Fore.BLUE)
        print("from accelerate import Accelerator")
        print("from torch.optim import AdamW")
        print("from transformers import AutoModelForSequenceClassification, get_scheduler")
        print(Fore.GREEN + "# Instantiates an Accelerator object that will look at the environment and initialize the proper distributed setup" + Fore.BLUE)
        print("accelerator = Accelerator() # Instantiates an Accelerator object that will look at the environment and initialize the proper distributed setup")
        print(Fore.BLUE + "model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)" + Style.RESET_ALL)
        print("optimizer = AdamW(model.parameters(), lr=3e-5)")
        print(Fore.GREEN + "# Wrap the dataloaders, model, and optimizer in the accelerator container to enable distributed training" + Fore.BLUE)
        print("train_dl, eval_dl, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer) # no `to(device)` is needed later on")
        print("num_epochs = 3")
        print("num_training_steps = num_epochs * len(train_dl)")
        print("lr_scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)")
        print("progress_bar = tqdm(range(num_training_steps))")
        print("model.train()")
        print("for epoch in range(num_epochs):")
        print("\tfor batch in train_dl:")
        print("\t\toutputs = model(**batch)")
        print("\t\tloss = outputs.loss")
        print("\t\taccelerator.backward(loss) # the last alteration required")
        print("\t\toptimizer.step()")
        print("\t\tlr_scheduler.step()")
        print("\t\toptimizer.zero_grad()")
        print("\t\tprogress_bar.update(1)")
        print(Style.RESET_ALL)
        input("And this is it for the training (without evaluation) with distributed training! Press Enter to continue...")

    # if yes("Save final checkpoint? (Y/N) "):
    #     path = input("Directory (default ./finetuned): ").strip() or "./finetuned"
    #     if use_accelerate:
    #         accelerator.wait_for_everyone()
    #         unwrapped = accelerator.unwrap_model(model)
    #         save_to_dir(unwrapped, bundle["tokenizer"], path)
    #     else:
    #         save_to_dir(model, bundle["tokenizer"], path)

    # print("🏁 Training finished.  "
    #       "Plot the logged losses in W&B or another tool to inspect "
    #       "learning curves (cf. 3_4.txt).")


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
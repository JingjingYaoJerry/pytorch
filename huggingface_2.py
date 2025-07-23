"""
huggingface_2.py
by Jingjing YAO (Jerry)

An interactive command-line utility that wraps common workflows of the
🤗 Transformers library.  Users can:

* run high-level pipelines,
* manually combine tokenizers and models,
* dynamically import any model / tokenizer class by string,
* save checkpoints locally and optionally push them to the Hub,
* inspect logits / hidden states / etc. for debugging,
* run demos for examples in the tutorial.

The script is entirely self-contained; no external HTML, images, or other
assets are required.
"""

import os
import sys
import importlib
import pathlib
from typing import Optional

import torch


# ----------------------------- helpers ----------------------------- #
def dynamic_import(class_name: str):
    """
    Dynamically import a single module from the ``transformers`` package.
    (Equivalent to ``from transformers import <class_name>``.)

    Parameters
    ----------
    class_name : str
        The exact symbol to import (e.g. ``"AutoModelForSequenceClassification"``).

    Returns
    -------
    type
        The class (or function) object requested.

    Raises
    ------
    ValueError
        If the requested symbol does not exist inside ``transformers``.
    """
    try:
        return getattr(__import__("transformers", fromlist=[class_name]),
                       class_name)
    except AttributeError as exc:
        raise ValueError(f"{class_name!r} is not available in "
                         "transformers") from exc


def yes(prompt: str) -> bool:
    """
    Helper function for yes/no prompts.

    Parameters
    ----------
    prompt : str
        The question displayed to the user.

    Returns
    -------
    bool
        ``True`` if the user types ``Y`` or ``y``, otherwise
        ``False``.
    """
    return input(prompt).strip().upper() == 'Y'


def save_to_dir(model, tokenizer, directory: str) -> None:
    """
    Save a model (and optionally its tokenizer) locally via `save_pretrained`.

    Parameters
    ----------
    model :
        Any ``PreTrainedModel`` instance.
    tokenizer :
        A matching tokenizer or ``None``.
    directory : str
        Target directory.  It is created if necessary.
    """
    os.makedirs(directory, exist_ok=True) # no exception if dir exists
    model.save_pretrained(directory)
    if tokenizer is not None:
        tokenizer.save_pretrained(directory)
    print(f"✓ Model{' & tokenizer' if tokenizer else ''} saved to {directory}")


def maybe_push_to_hub(model, tokenizer) -> None:
    """
    Optionally upload the current checkpoint to the Hugging Face Hub.

    The function asks the user for confirmation, performs authentication
    (either notebook or CLI), and calls :py:meth:`~transformers.PreTrainedModel.push_to_hub`.

    Parameters
    ----------
    model :
        The model to push.
    tokenizer :
        The corresponding tokenizer, or ``None``.
    """
    if not yes("Push this checkpoint to the Hugging Face Hub? (Y/N) "):
        return
    try:
        from huggingface_hub import notebook_login
    except ModuleNotFoundError:
        print("⚠️  The package `huggingface_hub` is not installed; "
              "run `pip install huggingface_hub` first.")
        return

    if yes("Notebook environment? (Y/N) "):
        notebook_login()
    else:
        print("Run `huggingface-cli login` in your terminal "
              "before pushing.")

    repo_name = input("Repository name under your namespace "
                      "(e.g. my-awesome-model): ").strip()
    try:
        model.push_to_hub(repo_name)
        if tokenizer is not None:
            tokenizer.push_to_hub(repo_name)
        print(f"✓ Pushed to https://huggingface.co/<username>/{repo_name}")
    except Exception as e:
        print(f"Push failed: {e}")


# --------------------------- main logic ---------------------------- #
def main():
    """
    Entry point of the script.

    The user first chooses between:

    1. Loading an *existing* local checkpoint.
    2. Running a high-level pipeline workflow.
    3. Running a manual tokenizer+model workflow.

    The corresponding helper function is invoked.
    """
    print("By default, any pre-trained Transformer models & tokenizers are saved under ~/.cache/huggingface/hub/.")
    print(f"i.e., {pathlib.Path.home() / '.cache' / 'huggingface' / 'hub'}")
    print("It will prevent re-downloading the same checkpoints, but you should still use the same method to load them.")
    print("Hence, unless there were custom configs/params/etc. saved locally, pick 'N' here!")
    if yes("Load model/tokenizer from a local directory? (Y/N) "):
        local_dir = input("Enter directory (default ./saved): ").strip() \
                   or "saved" # the default directory in the current folder
        if not pathlib.Path(local_dir).exists():
            print("Directory does not exist.")
            sys.exit(1)

        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModel.from_pretrained(local_dir)
        print("✓ Loaded model & tokenizer from local disk.")
        inspect_and_exit(model, tokenizer)
        return  # finished

    # ----------------------------------------------------------------
    preference = input(
        "Any preference between using a pipeline or a "
        "model with a tokenizer? (pipeline/tokenizer_model_post): "
    ).strip().lower()

    if preference not in {'pipeline', 'tokenizer_model_post'}:
        preference = 'pipeline' if \
            not yes('Got a specific checkpoint? (Y/N) ') else \
            'tokenizer_model_post'

    if preference == 'pipeline':
        pipeline_mode()
    else:
        tokenizer_model_post_mode()


# ------------------------- pipeline branch ------------------------- #
def pipeline_mode():
    """
    Interactive wrapper around `transformers.pipeline`.

    The function:

    * lets the user choose a task,
    * gathers prompts / inputs,
    * constructs a pipeline instance,
    * runs inference and prints the results.
    """
    from transformers import pipeline

    print("💡 Pipeline mode selected.")
    pipeline_tasks = {
        'Text': [
            'text-generation', 'text-classification', 'summarization',
            'translation', 'zero-shot-classification', 'feature-extraction',
            'ner', 'fill-mask', 'question-answering', 'sentiment-analysis',
        ],
        'Image': [
            'image-to-text', 'image-classification', 'object-detection'
        ],
        'Audio': [
            'automatic-speech-recognition', 'audio-classification',
            'text-to-speech',
        ],
        'Multimodal': ['image-text-to-text']
    }

    task = input("Enter pipeline task (e.g., text-generation, image-classification): ").strip()
    # Check if the task is in the <b>flattened</b> list of tasks
    if task not in sum(pipeline_tasks.values(), []):
        print("The task specified was not part of the instruction.")
        sys.exit(1)

    # Collect the "most basic" parameters based on the task type
    if task not in pipeline_tasks['Text']:
        print("Due to the nature of the task, 'contents' will be collected instead of 'prompts'.")
        contents = []
    elif task == 'question-answering':
        print("Due to the nature of the task, 'contexts' and 'questions' will be collected instead of 'prompts'.")
        contexts = []
        questions = []
    else:
        print("No special instruction was given, hence 'prompts' will be collected.")
        prompts = []

    while True:
        if task == 'fill-mask':
            prompts.append(input("Prompt with mask token (e.g., <mask> | [MASK]): "))
        elif task == 'question-answering':
            contexts.append(input("Context: "))
            questions.append(input("Question: "))
        elif task in pipeline_tasks['Text']:
            prompts.append(input("Prompt: "))
        else:
            contents.append(input("Content (file path / url): "))
        if not yes("More request? (Y/N) "):
            break

    model_ckpt = input("Pipeline model's checkpoint (leave blank for default): ").strip() or None
    pl = pipeline(task, model=model_ckpt)

    # Collect the correspondingly required parameters based on the task and call the pipeline
    if task == 'zero-shot-classification':
        labels = input("Candidate labels (comma separated): ").split(", ")
        result = pl(prompts, candidate_labels=labels)
    elif task in {'text-generation', 'summarization', 'translation'}:
        num_return_sequences = int(input("Number of return sequences: "))
        min_len = int(input("Min length: "))
        max_len = int(input("Max length: "))
        result = pl(prompts, num_return_sequences=num_return_sequences,
                    min_length=min_len, max_length=max_len)
    elif task == 'fill-mask':
        top_k = int(input("Top-k candidates: "))
        result = pl(prompts, top_k=top_k)
    elif task == 'ner':
        grouped_entities = yes("Group sub-entities (against sub-entities)? (Y/N) ")
        result = pl(prompts, grouped_entities=grouped_entities)
    elif task == 'question-answering':
        result = pl(question=questions, context=contexts)
    elif task in pipeline_tasks['Text']:
        result = pl(prompts)
    else:
        result = pl(contents)

    print("=== Pipeline outputs ===")
    print(result)


# -------------------- tokenizer + model branch --------------------- #
def tokenizer_model_post_mode():
    """
    Manual workflow that mirrors the three-step pipeline
    in `Behind the pipeline` & `Models`:

    1. Pre-processing with a tokenizer.
    2. Forward pass through a (possibly task-specified) model.
    3. Post-processing logits with the corresponding loss function.

    The user can dynamically pick tokenizer & model classes, change
    checkpoints, save to disk, and push to the Hub.
    """
    raw_inputs = [
        "Hello world!", "This is a test.", "Transformers are great!",
    ]

    #============ max_sequence_length Demo ============#
    if yes("According to the tutorial, it was suggested truncating the raw inputs straight " \
    "considering a model's accepted max length; would you like to do so? (Y/N) "):
        max_sequence_length = int(input("Enter max_sequence_length: ").strip())
        for raw_input in raw_inputs:
            if len(raw_input) > max_sequence_length:
                print(f"Below sequence's length exceeds max_sequence_length: ")
                print(raw_input)
                print("It will be truncated by [:max_sequence_length].")
                raw_input = raw_input[:max_sequence_length]
            print(f"Processed input: \n{raw_input}")
        print("Truncation complete.")
        input("Press Enter to continue...")

    print("Do note that a model's corresponding tokenizer returns additional inputs (other than input_ids, etc.) " \
          "accepted by this model.")
    print("Hence, using the same checkpoint for both tokenizer and model is recommended.")
    checkpoint = input(
        "Tokenizer checkpoint "
        "(blank for distilbert-base-uncased-finetuned-sst-2-english): "
    ).strip() or "distilbert-base-uncased-finetuned-sst-2-english"

    # ==================== 1. Tokenizer ====================== #
    print("Models can only process numbers, and hence a tokenizer is needed beforehand!")
    print("Three types of tokenizers' algorithms:")
    print("1. Word-Based (e.g., with split()) with [UNK] or <unk> => Large Vocabulary Size Possibly with Many OOV Tokens")
    print("2. Character-Based (e.g., with character-level CNNs) => Little Vocabulary Size and OOV Tokens " \
          "BUT Long Sequences to be Processed with Less Meaningful Tokens (according to the language)")
    print("3. Subword-Based (e.g., with Byte-Pair Encoding) with the Principle of " \
          "'Keeping Frequently Used Words and Decomposing Rare Words' => " \
          "Good Convergence with Small Vocabularies and Little OOV Tokens")
    if yes('Use a specific tokenizer architecture (i.e., Algorithm for Tokenization & Vocabulary for Mapping)? (Y/N) '):
        tok_class = input("Tokenizer class name: ").strip()
        TokenizerClass = dynamic_import(tok_class)
    else:
        from transformers import AutoTokenizer as TokenizerClass
    tokenizer = TokenizerClass.from_pretrained(checkpoint)

    #============ Tokenization Demo ============#
    if yes('Explore the tokenizer separately? (Y/N) '):
        try:
            sequence = "Tokenizer's process is simple"
            tokens = tokenizer.tokenize(sequence)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            print(f"Raw sequence: {sequence}")
            print(f"After going through this tokenizer's tokenization: {tokens}")
            print(f"After converting tokens to IDs: {ids}")
        except Exception as e:
            print(f"Demo failed: {e}")

    encoded_inputs = tokenizer(
        raw_inputs, return_tensors="pt"
    )
    print(f"Inputs' structure: "
          f"input_ids: tensor(n_batch, seq_len), token_type_ids: tensor(n_batch, seq_len), attention_mask: tensor(n_batch, seq_len)")
    print(f"Token Type IDs (Indications of different sentences): {encoded_inputs['token_type_ids']}")
    print(f"Attention Mask (Indications of attended and not attended tokens): {encoded_inputs['attention_mask']}")

    #============ Decoding Demo ============#
    if yes('Decode to see the automatically added special tokens (some models may not need special tokens)? (Y/N) '):
        try:
            print("Decoding the input IDs to see the original text with special tokens:")
            decoded_inputs = tokenizer.batch_decode(encoded_inputs['input_ids'])
            print("Decoded inputs:", decoded_inputs)
        except Exception as e:
            print(f"Demo failed: {e}")
    
    #============ Padding Demo ============#
    if yes('Any seq_len difference? (Y/N) '):
        print('To have rectangular tensors for matrix operations, padding is needed!')
        encoded_inputs = tokenizer(
            raw_inputs, padding=True, return_tensors="pt"
        )
        print("According to 'tokenizer.pad_token_id'...")
        print(f"Input IDs of {tokenizer.pad_token_id} were padded to shorter sequences (as they are not 'analyzed' by the model): ")
        print(f"Input IDs: {encoded_inputs['input_ids']}")

    #============ Truncation Demo ============#
    print("Tensors might get too long to be processed by the model, and hence truncation is needed!")
    if yes("Truncate to a specific length with tokenizer's truncation parameter? (Y/N) "):
        max_len = int(input("Enter max length (better check model's accepted max length): ").strip())
        encoded_inputs = tokenizer(
            raw_inputs, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        print("Inputs encoded with a max_length of", max_len)
    else:
        print(f"Tensors will be truncated to the maximum length accepted by the model (i.e., "
              f"model.config.max_position_embeddings == {model.config.max_position_embeddings}).")
        encoded_inputs = tokenizer(
            raw_inputs, padding=True, truncation=True, return_tensors="pt"
        )
        print("Inputs encoded without a max_length")

    #============ Truncation Demo ============#
    print("Do note that some model may accept inputs in a different type than 'pt' (e.g., 'np' or 'tf')...")
    print("Which can be specified in the tokenizer's return_tensors parameter.")
    input("Press Enter to continue...")

    print(f"Input IDs to be processed by the model: {encoded_inputs['input_ids']}")
    
    # ==================== 2. Model ====================== #
    model_ckpt = checkpoint
    print("Do note that a model's corresponding tokenizer returns additional inputs (other than input_ids, etc.) " \
          "accepted by this model.")
    print("Hence, using the same checkpoint for both tokenizer and model is recommended.")
    if yes('Different checkpoint for the model? (Y/N) '):
        model_ckpt = input("Model checkpoint (leave empty for the same): ").strip() or model_ckpt

    #============ Batching Demo ============#
    if yes('Can transformer models process a single input? (Y/N) '):
        print("Not really, transformer models are designed to process batches of inputs.")
        print("If a single input is given, an IndexError will be raised:")
        try:
            from transformers import AutoModelForSequenceClassification
            demo_model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            demo_sequence = "Hello world!"
            demo_tokens = tokenizer.tokenize(demo_sequence)
            demo_ids = tokenizer.convert_tokens_to_ids(demo_tokens)
            demo_input_ids = torch.tensor(demo_ids)
            demo_model(demo_input_ids)
        except IndexError as e:
            print(f"IndexError: {e}")
    print("However, it should be noted that a single input can still be processed with a Transformer tokenizer (i.e., a pipeline)...")
    print("Which by design batches a tensor with shape (1, seq_len) to mimic a batch of size 1 (i.e., batching).")
    input("Press Enter to continue...")

    if yes('Use a specific *architecture* (e.g. BertModel)? (Y/N) '):
        print("Raw unnormalized logits will be outputted for postprocessing with a specific task head!")
        mdl_class_name = input("Model architecture: ").strip()
        ModelClass = dynamic_import(mdl_class_name)
        model = ModelClass.from_pretrained(model_ckpt)
        outputs = model(**encoded_inputs)
        print("Outputs:", outputs)
    else:
        if yes('Use a specific task head (AutoModelFor* …)? (Y/N) '):
            print("Raw unnormalized logits will be outputted for postprocessing with a specific task head!")
            head_class = input("Model type: ").strip()
            ModelClass = dynamic_import(head_class)
            model = ModelClass.from_pretrained(model_ckpt)
            # To output the corresponding logits for the specific task
            outputs = model(**encoded_inputs)
            print(f"Output structure for AutoModel* (with a specific task head): "
                  f"logits (for classification): tensor(n_batch, n_classes), loss: {outputs['loss'].type()}, hidden_states: {outputs['hidden_states'].type()}, attentions: {outputs['attentions'].type()}"
                  "OR start_logits and end_logits (for question answering), etc.")
            print(f"Outputs: {outputs}")
            print(f"Logits: {outputs ['logits']}")
            print(f"The positional labels for the logits: {model.config.id2label}")
        else:
            print("The last hidden states will be outputted instead of logits without a specific task head!")
            from transformers import AutoModel as ModelClass # outputs the hidden states
            model = ModelClass.from_pretrained(model_ckpt)
            outputs = model(**encoded_inputs)
            print(f"Output structure for AutoModel (with no specified task head): "
                  f"last_hidden_state: tensor(n_batch, seq_len, hidden_size), pooler_output: tensor(n_batch, hidden_size)")

    #============ Attention Mask Demo ============#
    if yes('Attention masks were inputed to the model as well; do they affect the outputs? (Y/N) '):
        pass
    else:
        print("Actually they do! Attention masks enable the model to ignore the padded tokens, " \
              "leaving the output unchanged against an unpadded input!")
    print("Do recall padding's necessity for rectangular tensors...")
    input("Press Enter to continue...")

    # ==================== 3. Postprocessing ====================== #
    #============ Postprocessing Demo ============#
    try:
        preds = torch.nn.functional.softmax(outputs['logits'], dim=-1)
        print("Softmax predictions:", preds)
        print("Label mapping:", model.config.id2label)
    except (KeyError, AttributeError):
        print("No logits found; probably loaded a base model.")

    #============ Save & Push ============#
    if yes("Save this model/tokenizer locally? (Y/N) "):
        print("Two files are saved for the model -- config.json (attributes for the architecture) & pytorch_model.bin (state dict)")
        print("For the tokenizer, two files are saved for the algorithm and the vocabulary")
        save_dir = input("Directory to save (default ./saved): ").strip() \
                   or "saved" # the default directory in the current folder
        save_to_dir(model, tokenizer, save_dir)

    maybe_push_to_hub(model, tokenizer)

    inspect_and_exit(model, tokenizer)


# ---------------------- utilities for demo ------------------------- #
def inspect_and_exit(model, tokenizer: Optional[object] = None):
    """
    Tiny interactive inspector executed at the very end of the workflow.

    It lets the user type an arbitrary sentence, feeds it through the
    supplied ``tokenizer`` and ``model``, then prints either logits +
    probabilities (for models with classification heads) or the shape of
    the last hidden state (for base models).

    Parameters
    ----------
    model :
        The Transformer model to use for inference.
    tokenizer : object, optional
        The associated tokenizer.
    """
    if not yes("Run a quick inference test before exiting? (Y/N) "):
        return
    sentence = input("Enter a sentence: ").strip() or "Hello world!"
    toks = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model(**toks)

    if hasattr(out, "logits"):
        probs = torch.nn.functional.softmax(out.logits, dim=-1)
        print("Logits:", out.logits)
        try:
            id2label = model.config.id2label
            labels = [id2label[i] for i in range(len(id2label))]
            print("Probabilities:", dict(zip(labels, probs[0].tolist())))
        except Exception:
            print("Softmax probabilities:", probs)
    else:
        print("Last hidden state shape:", out.last_hidden_state.shape)

    print("👋  Done.  Goodbye!")


# ------------------------------- run ------------------------------- #
if __name__ == "__main__":
    main()
import importlib
import torch

preference = input("Any preference between using a pipeline or a model with a tokenizer? "
                   "(pipeline/tokenizer_model_post): "
                   ).strip().lower()
if preference not in ['pipeline', 'tokenizer_model_post']:
    if input('Got a specific checkpoint? (Y/N) ').strip().upper() == 'N':
        method = 'pipeline'
    else:
        method = 'tokenizer_model_post'

if method == 'pipeline':
    from transformers import pipeline
    print("Using pipeline method")
    pipeline_tasks = {
        'Text': [
            'text-generation',
            'text-classification',
            'summarization',
            'translation',
            'zero-shot-classification',
            'feature-extraction', 
            'ner', 
            'fill-mask', 
            'question-answering', 
            'sentiment-analysis', 
        ], 
        'Image': [
            'image-to-text', 
            'image-classification', 
            'object-detection', 
        ], 
        'Audio': [
            'automatic-speech-recognition', 
            'audio-classification', 
            'text-to-speech', 
        ], 
        'Multimodal': [
            'image-text-to-text', 
        ]
    }

    pipeline_task = input("Enter pipeline task (e.g., text-generation, image-classification): ")

    if pipeline_task not in pipeline_tasks['Text']:
        contents = []
    elif pipeline_task == 'question-answering':
        contexts = []
        questions = []
    else:
        prompts = []

    more = True
    while more:

        if pipeline_task == 'fill-mask':
            prompt += input("Enter prompt with the model's specified mask token (e.g., The capital of France is <mask> or [MASK]): ")
        elif pipeline_task == 'question-answering':
            context += input("Enter context: ")
            question += input("Enter question: ")
        elif pipeline_task not in pipeline_tasks['Text']:
            content += input("Enter content (e.g., image path, audio file path): ")
        else:
            prompt += input("Enter prompt: ")

        more = (input("More than one request? (Y/N) ")).strip().upper() == 'Y'

    model = input("Enter model name (checkpoints / parameters): ")

    pipeline_instance = pipeline(pipeline_task, model=model if model else None, )

    if pipeline_task == 'zero-shot-classification':
        labels = input("Enter candidate labels (comma-separated): ").split(', ')
        outputs = pipeline_instance(prompt, candidate_labels=labels)
    elif pipeline_task == 'text-generation' or pipeline_task == 'summarization' or pipeline_task == 'translation':
        num_return_sequences = int(input("Enter number of return sequences: "))
        min_length = int(input("Enter minimum length: "))
        max_length = int(input("Enter max length: "))
        outputs = pipeline_instance(prompt, min_length=min_length, max_length=max_length, num_return_sequences=num_return_sequences)
    elif pipeline_task == 'fill-mask':
        top_k = int(input("Enter number of top candidates to return: "))
        outputs = pipeline_instance(prompt, top_k=top_k)
    elif pipeline_task == 'ner':
        grouped_entities = (input("Group entities (against sub-entities)? (True/False): ")).strip().lower().capitalize() == 'True'
        outputs = pipeline_instance(prompt, grouped_entities=grouped_entities)
    elif pipeline_task == 'question-answering':
        outputs = pipeline_instance(question=question, context=context)
    elif pipeline_task not in pipeline_tasks['Text']:
        outputs = pipeline_instance(content)
    else:
        outputs = pipeline_instance(prompt)

    print(outputs)

elif method == 'tokenizer_model_post':
    checkpoint = input("Enter checkpoint (e.g., distilbert-base-uncased-finetuned-sst-2-english): ").strip() or "distilbert-base-uncased-finetuned-sst-2-english"
    if input('Use a specific tokenizer architecture? (Y/N) ').strip().upper() == 'Y':
        specified_tokenizer = input("Enter tokenizer class "
        "(e.g., AutoTokenizer, BertTokenizer, RobertaTokenizer, GPT2Tokenizer, T5Tokenizer, XLMRobertaTokenizer): "
        ).strip()
        try:
            # from transformers import <specified>
            TokenizerClass = getattr(
                __import__("transformers", fromlist=[specified_tokenizer]),
                specified_tokenizer
            )
        except AttributeError:
            raise ValueError(f"{specified_tokenizer!r} 不是 transformers 中可以导入的模型类")
        tokenizer = TokenizerClass.from_pretrained(checkpoint)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # To split raw text into words/subwords/symbols (i.e., tokens) with special tokens
    # To map tokens to their corresponding integer IDs in the pre-trained vocabulary
    # To add additional inputs (e.g., attention masks)
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(f"Inputs' structure: "
          f"input_ids: tensor(n_batch, seq_len), attention_mask: tensor(n_batch, seq_len)"
          )
    
    # Check whether using the same checkpoint for both tokenizer and model
    if input('Use a different checkpoint for model? (Y/N) ').strip().upper() == 'Y':
        checkpoint = input("Enter model checkpoint (e.g., distilbert-base-uncased-finetuned-sst-2-english): ").strip() or "distilbert-base-uncased-finetuned-sst-2-english"
    
    if input('Use a specific model architecture? (Y/N) ').strip().upper() == 'Y':
        print("Raw unnormalized logits will be outputted for postprocessing with a specific task head!")
        specified_model = input("Enter model architecture "
        "(e.g., BertModel, RobertaModel, GPT2Model, T5Model, XLMRobertaModel): "
        ).strip()
        try:
            # from transformers import <specified>
            ModelClass = getattr(
                __import__("transformers", fromlist=[specified_model]),
                specified_model
            )
        except AttributeError:
            raise ValueError(f"{specified_model!r} 不是 transformers 中可以导入的模型类")
        model = ModelClass.from_pretrained(checkpoint)
        outputs = model(**inputs)

    else:
        if input("Use a specific model type? (Y/N) ").strip().upper() == 'Y':
            print("Raw unnormalized logits will be outputted for postprocessing with a specific task head!")
            specified = input("Enter model type "
            "(e.g., AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForMultipleChoice, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoModelForTokenClassification): "
            ).strip()
            try:
                # from transformers import <specified>
                ModelClass = getattr(
                    __import__("transformers", fromlist=[specified]),
                    specified
                )
            except AttributeError:
                raise ValueError(f"{specified!r} 不是 transformers 中可以导入的模型类")
            
            model = ModelClass.from_pretrained(checkpoint)
            # To output the corresponding logits for the specific task
            outputs = model(**inputs)
            print(f"Output structure for AutoModel* (with a specific task head): "
                  f"logits (for classification): tensor(n_batch, n_classes), loss: {outputs['loss'].type()}, hidden_states: {outputs['hidden_states'].type()}, attentions: {outputs['attentions'].type()}"
                  "OR start_logits and end_logits (for question answering), etc.")
            print(f"Outputs: {outputs}")
            print(f"Logits: {outputs ['logits']}")
            print(f"The positional labels for the logits: {model.config.id2label}")
        else:
            print("The last hidden states will be outputted instead of logits without a specific task head!")
            from transformers import AutoModel # outputs the hidden states
            model = AutoModel.from_pretrained(checkpoint)
            outputs = model(**inputs)
            print(f"Output structure for AutoModel (with no specified task head): "
                  f"last_hidden_state: tensor(n_batch, seq_len, hidden_size), pooler_output: tensor(n_batch, hidden_size)"
                  )
            # Appending the corresponding head to acomplish the specific task

# Postprocessing the outputs of the model
try:
    predictions = torch.nn.functional.softmax(outputs['logits'], dim=-1)
    print(f"Predictions: {predictions}")
except KeyError:
    print("No logits found in the outputs. The model may not have a classification head.")
   
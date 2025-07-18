from transformers import pipeline

pipeline_instance = pipeline('summarization')
outputs = pipeline_instance("Hugging Face is creating a tool that democratizes AI for everyone.", num_return_sequences=2, max_length=20)
print(outputs)
exit()

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


if pipeline_task == 'fill-mask':
    prompt = input("Enter prompt with the model's specified mask token (e.g., The capital of France is <mask> or [MASK]): ")
elif pipeline_task == 'question-answering':
    context = input("Enter context: ")
    question = input("Enter question: ")
elif pipeline_task not in pipeline_tasks['Text']:
    content = input("Enter content (e.g., image path, audio file path): ")
else:
    prompt = input("Enter prompt: ")

model = input("Enter model name (checkpoints / parameters): ")

pipeline_instance = pipeline(pipeline_task, model=model if model else None, )

if pipeline_task == 'zero-shot-classification':
    labels = input("Enter candidate labels (comma-separated): ").split(', ')
    outputs = pipeline_instance(prompt, candidate_labels=labels)
elif pipeline_task == 'text-generation' or pipeline_task == 'summarization' or pipeline_task == 'translation':
    num_return_sequences = int(input("Enter number of return sequences: "))
    min_length = int(input("Enter minimum length: "))
    max_length = int(input("Enter max length: "))
    outputs = pipeline_instance(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
elif pipeline_task == 'fill-mask':
    top_k = int(input("Enter number of top predictions to return: "))
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
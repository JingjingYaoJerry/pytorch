from transformers import pipeline

classifier = pipeline("sentiment-analysis")
outputs = classifier("I've been waiting for a HuggingFace course my whole life.")
print(outputs)
tasks = [

]

pipeline_tasks = {
    'Text': [
        'text-generation',
        'text-classification',
        'summarization',
        'translation',
        'zero-shot-classification',
        'feature-extraction', 
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
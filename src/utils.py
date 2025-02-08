import os

def load_text(file_path):
    """
    Loads text from a file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def save_text(file_path, text):
    """
    Saves text to a file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

import json
from datasets import Dataset

def load_custom_dataset(file_path):
    """
    Loads a custom dataset from a JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to the format expected by the Trainer API
    formatted_data = []
    for entry in data:
        context = entry["context"]
        question = entry["question"]
        answer = entry["answer"]
        
        # Find the start and end positions of the answer in the context
        start = context.find(answer)
        end = start + len(answer)
        
        formatted_data.append({
            "context": context,
            "question": question,
            "answers": {"text": [answer], "answer_start": [start]}
        })
    
    return Dataset.from_list(formatted_data)
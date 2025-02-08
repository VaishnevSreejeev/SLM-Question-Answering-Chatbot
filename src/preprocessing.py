import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)

def preprocess_text(text):
    """
    Cleans the text by removing special characters, extra spaces, and converting to lowercase.
    """
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()
    return text

def split_into_chunks(text, max_chunk_size=512):
    """
    Splits the text into chunks of approximately `max_chunk_size` tokens, ensuring sentence boundaries.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
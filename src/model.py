from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('./models/final_model')
model = AutoModelForQuestionAnswering.from_pretrained('./models/final_model')

def generate_answer(context_chunks, question):
    """
    Generates an answer for a given question and list of context chunks.
    """
    best_answer = ""
    best_score = -float('inf')

    for chunk in context_chunks:
        inputs = tokenizer.encode_plus(
            question,
            chunk,
            max_length=512,
            truncation="only_second",
            padding="max_length",
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Find the most probable answer span
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores) + 1
        answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer_tokens))

        # Skip invalid answers (e.g., [CLS], empty strings, or repeated question text)
        if answer.strip() == "[CLS]" or not answer.strip() or question.lower() in answer.lower():
            continue

        # Calculate confidence score
        score = start_scores[0][answer_start] + end_scores[0][answer_end - 1]
        if score > best_score:
            best_answer = answer
            best_score = score

    # Return fallback message if no valid answer is found
    return best_answer.strip() if best_answer else "No answer found."
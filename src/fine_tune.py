from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from utils import load_custom_dataset

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')

def preprocess_function(examples):
    """
    Tokenizes the dataset for fine-tuning.
    Handles batches of examples when batched=True.
    """
    # Extract lists of contexts, questions, and answers from the batch
    contexts = examples["context"]
    questions = examples["question"]
    answers = examples["answers"]

    # Tokenize the inputs
    inputs = tokenizer(
        questions,
        contexts,
        max_length=512,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    # Map tokenized answers to input IDs
    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # Find the token indices corresponding to the answer
        token_start = token_end = None
        for idx, (start, end) in enumerate(offsets):
            if start <= start_char < end:
                token_start = idx
            if start <= end_char <= end:
                token_end = idx
                break

        start_positions.append(token_start)
        end_positions.append(token_end)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

if __name__ == "__main__":
    # Load the custom dataset
    file_path = r'slm-question-answering\data\custom_dataset.json'
    dataset = load_custom_dataset(file_path)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./models/checkpoints',  # Directory to save checkpoints
        eval_strategy="epoch",             # Evaluate after each epoch
        learning_rate=2e-5,                # Learning rate
        per_device_train_batch_size=8,    # Batch size for training
        per_device_eval_batch_size=8,     # Batch size for evaluation
        num_train_epochs=3,               # Number of training epochs
        weight_decay=0.01,                # Weight decay for regularization
        logging_dir='./logs',             # Directory for logs
        logging_steps=10,                 # Log every 10 steps
        save_steps=500,                   # Save checkpoint every 500 steps
        save_total_limit=2,               # Limit the total number of checkpoints
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,  # Tokenized training dataset
        eval_dataset=tokenized_dataset,   # Tokenized evaluation dataset
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained('./models/final_model')
    tokenizer.save_pretrained('./models/final_model')
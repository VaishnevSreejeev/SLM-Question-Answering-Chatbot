

# SLM Question-Answering Chatbot


This project implements an **interactive question-answering (QA) chatbot** based on a book. It leverages advanced Natural Language Processing (NLP) techniques such as text preprocessing, chunk retrieval, and fine-tuned QA models to provide accurate answers to user queries.

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Customization](#customization)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

---

## Features

- **Interactive Chatbot Interface:**
  - Supports multiple interfaces:
    - Terminal-based chat.
    - Web-based chat using **Gradio**.
    - Web-based chat using **Streamlit**.
- **Fine-Tuned QA Model:**
  - Uses a pre-trained model (`deepset/roberta-base-squad2`) fine-tuned on custom datasets derived from the book.
- **Text Retrieval:**
  - Retrieves relevant chunks of text from the book using semantic similarity (powered by `sentence-transformers`).
- **Modular Design:**
  - Code is organized into reusable modules for easy customization and scalability.
- **Fallback Mechanism:**
  - Handles cases where no relevant chunk or answer is found.

---

## Installation

### Prerequisites
- Python 3.8 or higher.
- A modern web browser (for Gradio or Streamlit interfaces).

### Steps to Install
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/VaishnevSreejeev/SLM-Question-Answering-Chatbot
   cd slm-question-answering
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Book File:**
   - Place your book file in the `data/Raw/` directory.
   - Ensure the file is named `Book.txt`.
  
5. **Run the fine_tune.py:**
   -create slm-question-answering\models
   -the final model is to be stored in the same directory as slm-question-answering\models\final_model\model.safetensors ie in final_model

---

## Usage

### Gradio Web Interface
Launch the Gradio-based chatbot:
```bash
python src/run.py
```
- Open the provided local URL (e.g., `http://127.0.0.1:7860/`) in your browser.
- Interact with the chatbot through the web interface.

Sample Questions to ask



### Terminal-Based Chat -- Commented as of now
Run the terminal-based chatbot:
- Follow the prompts to ask questions about the book.
- Type `exit` to quit.

## Project Structure

The project is organized into modular components for clarity and maintainability:

```
slm-question-answering/
│
├── venv/                     # Virtual environment (ignored by Git)
│
├── data/                     # Store raw and processed data
│   ├── raw/                  # Raw book files (e.g., .txt or .pdf)
│   └── processed/            # Preprocessed text files
│
├── models/                   # Store trained models and checkpoints
│   ├── checkpoints/          # Intermediate model checkpoints
│   └── final_model/          # Final fine-tuned model
│
├── notebooks/                # Jupyter notebooks for experimentation (optional)
│   └── exploratory_analysis.ipynb
│
├── src/                      # Source code for the project
│   ├── preprocessing.py      # Text preprocessing functions
│   ├── fine_tune.py          # Retrieval pipeline
│   ├── model.py              # Model architecture and training logic
│   ├── retrieval.py          # Retrieval pipeline
│   ├── evaluation.py         # Evaluation metrics
│   └── utils.py              # Helper functions
│
├── tests/                    # Unit tests for your code (optional)
│   └── test_preprocessing.py
│
├── outputs/                  # Generated outputs (e.g., predictions, logs)
│   ├── predictions.json
│   └── logs/
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── run.py                    # Main script to execute the pipeline
```

---


## Customization

### Adding a New Book
1. Replace `data/Raw/Book.txt` with your new book file.
2. Re-run the preprocessing steps to generate chunks and embeddings.

### Fine-Tuning the QA Model
1. Update `data/custom_dataset.json` with new question-answer pairs.
2. Fine-tune the model again using `src/fine_tune.py`.

### Changing the Interface
- Modify `src/run_gradio.py` or `src/run_streamlit.py` to customize the UI.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Submit a pull request.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Transformers Library:** Thanks to Hugging Face for their amazing NLP tools.
- **Sentence Transformers:** Thanks to the creators of the `sentence-transformers` library for enabling efficient text embeddings.
- **Streamlit & Gradio:** Thanks for simplifying the creation of interactive web apps.

---

Feel free to customize this `README.md` further to suit your specific project needs. Let me know if you need help with anything else! 😊

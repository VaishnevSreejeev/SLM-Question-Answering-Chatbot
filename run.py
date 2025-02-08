
#Terminal based chat
"""

from src.preprocessing import preprocess_text, split_into_chunks
from src.retrieval import retrieve_relevant_chunk, get_embeddings
from src.model import generate_answer
from src.utils import load_text

def main():
    # Load and preprocess book
    try:
        book_text = load_text('slm-question-answering\data\Raw\Book.txt')  # Ensure correct path
        processed_text = preprocess_text(book_text)
        chunks = split_into_chunks(processed_text)
    except FileNotFoundError:
        print("Error: The file 'data/raw/book.txt' was not found. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"An error occurred while loading or preprocessing the book: {e}")
        return

    # Generate embeddings
    try:
        chunk_embeddings = get_embeddings(chunks)
    except Exception as e:
        print(f"An error occurred while generating embeddings: {e}")
        return

    print("Book loaded successfully. You can now ask questions!")
    
    # Interactive question-answering loop
    while True:
        try:
            # Get user input
            question = input("\nAsk: ").strip()
            if not question:
                print("Please enter a valid question.")
                continue

            # Retrieve relevant chunk
            relevant_chunk = retrieve_relevant_chunk(question, chunks, chunk_embeddings)
            if relevant_chunk is None:
                print("No relevant chunk found. Unable to answer the question.")
                continue

            # Split the relevant chunk into smaller chunks for the model
            relevant_chunks = split_into_chunks(relevant_chunk, max_chunk_size=512)
            answer = generate_answer(relevant_chunks, question)

            # Display the answer
            print(f"Answer: {answer}")

        except Exception as e:
            print(f"An error occurred while processing the question: {e}")

        # Check if the user wants to continue
        continue_input = input("Continue? (y/n): ").strip().lower()
        if continue_input != 'y':
            print("Exiting...")
            break

if __name__ == "__main__":
    main()

"""
#Interactive chat using gradio

import gradio as gr
from src.preprocessing import preprocess_text, split_into_chunks
from src.retrieval import retrieve_relevant_chunk, get_embeddings
from src.model import generate_answer
from src.utils import load_text

# Load and preprocess the book
def load_book():
    try:
        book_text = load_text(r'slm-question-answering/data/Raw/Book.txt')  # Use raw string or forward slashes
        processed_text = preprocess_text(book_text)
        chunks = split_into_chunks(processed_text)
        chunk_embeddings = get_embeddings(chunks)
        return chunks, chunk_embeddings
    except FileNotFoundError:
        raise FileNotFoundError("Error: The file 'data/raw/book.txt' was not found. Please ensure the file exists.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading or preprocessing the book: {e}")

# Initialize global variables
chunks, chunk_embeddings = load_book()

# Chatbot response function
def respond(message, chat_history):
    try:
        # Retrieve relevant chunk
        relevant_chunk = retrieve_relevant_chunk(message, chunks, chunk_embeddings)
        if relevant_chunk is None:
            return "No relevant chunk found. Unable to answer the question.", chat_history
        
        # Split the relevant chunk into smaller chunks for the model
        relevant_chunks = split_into_chunks(relevant_chunk, max_chunk_size=512)
        answer = generate_answer(relevant_chunks, message)

        # Fallback for missing answers
        if not answer.strip():
            answer = "No answer found."

        # Update chat history
        chat_history.append((message, answer))
        return "", chat_history  # Clear input box and update chat history

    except Exception as e:
        return f"An error occurred: {e}", chat_history

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# Interactive QA Chatbot")
    gr.Markdown("Ask questions about the book, and the bot will provide answers!")

    # Chatbot component
    chatbot = gr.Chatbot(label="Conversation")
    message = gr.Textbox(label="Your Question", placeholder="Type your question here...")
    clear = gr.Button("Clear Conversation")

    # Define interactions
    message.submit(respond, [message, chatbot], [message, chatbot])  # Submit question
    clear.click(lambda: None, None, chatbot, queue=False)  # Clear chat history

# Launch the app
if __name__ == "__main__":
    demo.launch()




#chat using Streamlit code

"""
import streamlit as st
from src.preprocessing import preprocess_text, split_into_chunks
from src.retrieval import retrieve_relevant_chunk, get_embeddings
from src.model import generate_answer
from src.utils import load_text

# Load and preprocess the book
@st.cache_data  # Cache the data to avoid reloading
def load_book():
    try:
        book_text = load_text(r'slm-question-answering/data/Raw/Book.txt')  # Use raw string or forward slashes
        processed_text = preprocess_text(book_text)
        chunks = split_into_chunks(processed_text)
        chunk_embeddings = get_embeddings(chunks)
        return chunks, chunk_embeddings
    except FileNotFoundError:
        st.error("Error: The file 'data/raw/book.txt' was not found. Please ensure the file exists.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading or preprocessing the book: {e}")
        return None, None

# Initialize global variables
chunks, chunk_embeddings = load_book()

# Chatbot response function
def respond(message, chat_history):
    try:
        # Retrieve relevant chunk
        relevant_chunk = retrieve_relevant_chunk(message, chunks, chunk_embeddings)
        if relevant_chunk is None:
            return "No relevant chunk found. Unable to answer the question."
        
        # Split the relevant chunk into smaller chunks for the model
        relevant_chunks = split_into_chunks(relevant_chunk, max_chunk_size=512)
        answer = generate_answer(relevant_chunks, message)

        # Fallback for missing answers
        if not answer.strip():
            answer = "No answer found."

        # Update chat history
        chat_history.append((message, answer))
        return answer

    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit App
def main():
    st.title("Interactive QA Chatbot")
    st.markdown("Ask questions about the book, and the bot will provide answers!")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for user_message, bot_response in st.session_state.chat_history:
        st.text_area("You:", value=user_message, height=75, disabled=True)
        st.text_area("Bot:", value=bot_response, height=75, disabled=True)

    # User input
    user_input = st.text_input("Your Question:", placeholder="Type your question here...")
    if st.button("Send"):
        if user_input.strip():
            bot_response = respond(user_input, st.session_state.chat_history)
            st.session_state.chat_history.append((user_input, bot_response))
            st.experimental_rerun()  # Refresh the page to update the chat history

if __name__ == "__main__":
    main()

"""
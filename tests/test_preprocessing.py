from src.preprocessing import preprocess_text, split_into_chunks

def test_preprocess_text():
    text = "Hello, World! This is a TEST."
    expected = "hello world this is a test"
    assert preprocess_text(text) == expected

def test_split_into_chunks():
    text = "This is a sample text to split into chunks."
    chunks = split_into_chunks(text, chunk_size=3)
    assert len(chunks) == 2
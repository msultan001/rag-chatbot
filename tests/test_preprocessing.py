from src.data_preprocessing import clean_text

def test_clean_text_basic():
    """
    Test basic cleaning: lowercase and special character removal.
    """
    raw = "Hello World! 123"
    cleaned = clean_text(raw)
    assert cleaned == "hello world"

def test_clean_text_whitespace():
    """
    Test removal of extra whitespace.
    """
    raw = "  too    many   spaces  "
    cleaned = clean_text(raw)
    assert cleaned == "too many spaces"

def test_clean_text_non_string():
    """
    Test handling of non-string input.
    """
    assert clean_text(None) == ""
    assert clean_text(123) == ""

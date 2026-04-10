import re

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# utils/preprocessing.py

import spacy
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if token.like_num:
            continue
        if token.is_alpha or token.text.isalnum() or "-" in token.text:
            tokens.append(token.lemma_)
    return tokens

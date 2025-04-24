# tools/entity_extractor.py

import re

def extract_entities(text):
    """
    Extracts product codes like UX-09, CX-01, BX-11 from text.
    """
    return re.findall(r"\b(?:UX|BX|CX)-\d{2}\b", text.upper())

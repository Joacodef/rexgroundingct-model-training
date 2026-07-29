import re

# Adjective & hedge modifiers to strip
STRIP_MODIFIERS = [
    r'\b(stable|nonspecific|minimal|mild|moderate|severe|borderline|prominent|subcentimeter|tiny|small|large|focal|patchy)\b,?\s*',
    r',?\s*measuring\s+(?:approximately\s+|up\s+to\s+)?\d+(?:\.\d+)?\s*(?:mm|cm)[^,.]*',
    r'\b\d+(?:\.\d+)?\s*(?:mm|cm)\b\s*',
    r'\b(suggestive\s+of|consistent\s+with|likely\s+representing|concerning\s+for|suspicious\s+for)\b\s*',
]

def clean_finding_prompt(text: str) -> str:
    """
    Hybrid Medical Prompt Normalizer:
    Strips non-diagnostic clinical modifiers & measurements while preserving
    core finding entities and anatomical locations.
    """
    cleaned = text
    
    # 1. Strip measurement expressions
    cleaned = re.sub(r',?\s*measuring\s+(?:approximately\s+|up\s+to\s+)?\d+(?:\.\d+)?\s*(?:mm|cm)[^,.]*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b\d+(?:\.\d+)?\s*(?:mm|cm)\b', '', cleaned, flags=re.IGNORECASE)
    
    # 2. Strip quality & clinical hedge adjectives
    for pattern in STRIP_MODIFIERS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
    # 3. Clean trailing/leading spaces and punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'^[,\.\s]+|[,\.\s]+$', '', cleaned)
    
    if not cleaned.strip():
        return text.strip()
        
    return cleaned[0].upper() + cleaned[1:] if cleaned else text

if __name__ == "__main__":
    sample_prompts = [
        "Stable, nonspecific 6 mm subpleural nodule in the medial segment of the middle lobe",
        "Subcentimeter, minimal, nonspecific focal opacity in the posterobasal segment of the left lower lobe",
        "Patchy ground-glass opacities in both lungs",
        "Bilateral pleural effusion more prominent on the right, measuring 60 mm at its thickest point"
    ]
    
    print("=== HYBRID MEDICAL PROMPT NORMALIZER ===")
    for orig in sample_prompts:
        print(f"\nOriginal: '{orig}'")
        print(f"Cleaned : '{clean_finding_prompt(orig)}'")

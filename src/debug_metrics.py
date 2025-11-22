import evaluate
import nltk
import os

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

BASE_DIR = os.path.abspath(".")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = os.path.join(BASE_DIR, "data")

def debug_rouge():
    print("--- Debugging ROUGE Metrics ---")
    
    # Load Reference
    golden_path = os.path.join(DATA_DIR, "Golden_Sample.txt")
    with open(golden_path, 'r', encoding='utf-8') as f:
        reference = f.read()
    
    ref_len = len(reference.split())
    print(f"Reference Length: {ref_len} words")

    # Load Candidate (Gemma NoTAEG)
    cand_path = os.path.join(OUTPUT_DIR, "Consolidated_Narrative_Gemma-3-4B_NoTAEG.txt")
    with open(cand_path, 'r', encoding='utf-8') as f:
        candidate = f.read()
        
    cand_len = len(candidate.split())
    print(f"Candidate Length: {cand_len} words")
    
    print(f"Length Ratio: {cand_len / ref_len:.4f}")

    # Calculate ROUGE
    rouge = evaluate.load('rouge')
    
    # Standard Compute (F1 usually)
    results = rouge.compute(predictions=[candidate], references=[reference], use_aggregator=False)
    print(f"\nStandard ROUGE Results (F1):")
    print(results)
    
    # Let's try to manually estimate Recall
    ref_tokens = set(reference.lower().split())
    cand_tokens = set(candidate.lower().split())
    overlap = cand_tokens.intersection(ref_tokens)
    
    print(f"\nManual Unigram Analysis:")
    print(f"Unique Tokens in Ref: {len(ref_tokens)}")
    print(f"Unique Tokens in Cand: {len(cand_tokens)}")
    print(f"Overlap: {len(overlap)}")
    
    recall = len(overlap) / len(ref_tokens)
    precision = len(overlap) / len(cand_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print(f"Estimated Recall: {recall:.4f}")
    print(f"Estimated Precision: {precision:.4f}")
    print(f"Estimated F1: {f1:.4f}")

if __name__ == "__main__":
    debug_rouge()

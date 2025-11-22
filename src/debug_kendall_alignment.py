import os
import nltk
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Config
BASE_DIR = os.path.abspath(".")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CHRONOLOGY_PATH = os.path.join(DATA_DIR, "ChronologyOfTheFourGospels_PW.xml")
FILE_TO_TEST = os.path.join(OUTPUT_DIR, "Consolidated_Narrative_Gemma-3-4B.txt")

# Ensure NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_event_descriptions(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    descriptions = []
    for event_node in root.findall(".//event"):
        evt_id = event_node.get("id")
        desc_node = event_node.find("description")
        if desc_node is not None and desc_node.text:
            descriptions.append((evt_id, desc_node.text.strip()))
    return descriptions

def debug_alignment():
    print(f"--- Debugging Kendall's Tau Alignment for {os.path.basename(FILE_TO_TEST)} ---")
    
    # 1. Load Data
    events = get_event_descriptions(CHRONOLOGY_PATH)
    with open(FILE_TO_TEST, 'r', encoding='utf-8') as f:
        text = f.read()
    
    sentences = nltk.sent_tokenize(text)
    print(f"Loaded {len(events)} events and {len(sentences)} sentences from text.")

    # 2. Vectorize
    descriptions_text = [d[1] for d in events]
    vectorizer = TfidfVectorizer().fit(sentences + descriptions_text)
    
    sent_vectors = vectorizer.transform(sentences)
    
    print("\n--- Checking First 20 Events Alignment ---")
    print(f"{'Event ID':<10} | {'Expected Order':<15} | {'Matched Sentence Index':<25} | {'Similarity':<10} | {'Event Description (Truncated)'}")
    print("-" * 120)

    matches = []
    
    for i, (evt_id, desc) in enumerate(events):
        desc_vector = vectorizer.transform([desc])
        similarities = cosine_similarity(desc_vector, sent_vectors).flatten()
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Only consider it a match if score > 0.15 (same as metric script)
        if best_score > 0.15:
            matches.append((i, best_idx, best_score, evt_id, desc))
        
        # Print debug for first 20 events to see jumps
        if i < 20:
            status = "OK"
            if matches and len(matches) > 1:
                # Check if index went backwards compared to previous match
                prev_match_idx = matches[-2][1] if len(matches) >= 2 else -1
                if best_idx < prev_match_idx:
                    status = "BACKWARD <<"
            
            print(f"{evt_id:<10} | {i:<15} | {best_idx:<25} | {best_score:.4f}     | {desc[:40]}... {status if status != 'OK' else ''}")

    print("-" * 120)
    
    # Analyze Reversals
    reversals = 0
    for k in range(1, len(matches)):
        curr_sent_idx = matches[k][1]
        prev_sent_idx = matches[k-1][1]
        if curr_sent_idx < prev_sent_idx:
            reversals += 1
            if reversals <= 5: # Show first 5 reversals details
                curr_evt = matches[k]
                prev_evt = matches[k-1]
                print(f"\n[REVERSAL DETECTED]")
                print(f"  Previous: Event {prev_evt[3]} ('{prev_evt[4][:30]}...') found at Sentence {prev_evt[1]}")
                print(f"  Current:  Event {curr_evt[3]} ('{curr_evt[4][:30]}...') found at Sentence {curr_evt[1]}")
                print(f"  Reason: The metric thinks Event {curr_evt[3]} happened BEFORE Event {prev_evt[3]} in the text.")

    print(f"\nTotal Detected Matches: {len(matches)} / {len(events)}")
    print(f"Total Reversals (Out of Order detections): {reversals}")
    print("Conclusion: If 'Total Reversals' is high, the metric is confused by semantic similarity, preventing a 1.0 score.")

if __name__ == "__main__":
    debug_alignment()

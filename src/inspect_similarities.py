import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import nltk
from sentence_transformers import SentenceTransformer, util
from main import ChronologyParser, BibleXMLParser, FILES_CONFIG, DATA_DIR

def debug_similarities():
    print("--- Debugging Similarity Scores for BART ---")
    
    # 1. Load Data
    bible_parser = BibleXMLParser(DATA_DIR, FILES_CONFIG["gospels"])
    chrono_path = os.path.join(DATA_DIR, FILES_CONFIG["chronology"])
    chrono_parser = ChronologyParser(chrono_path, bible_parser)
    events = chrono_parser.get_events()
    
    # 2. Load BART Output
    bart_path = os.path.join("output", "Consolidated_Narrative_BART.txt")
    if not os.path.exists(bart_path):
        print("BART output not found.")
        return
        
    with open(bart_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    sentences = nltk.sent_tokenize(text)
    print(f"Loaded {len(events)} events and {len(sentences)} sentences from BART.")
    
    # 3. Load Model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 4. Compute Similarities
    event_descs = [e['description'] for e in events]
    
    print("Encoding...")
    event_embs = model.encode(event_descs, convert_to_tensor=True)
    sent_embs = model.encode(sentences, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(event_embs, sent_embs)
    
    # 5. Inspect Top Matches
    print("\n--- Top Matches Analysis ---")
    print(f"{'Event ID':<10} | {'Max Sim':<8} | {'Best Sentence (Truncated)'}")
    print("-" * 80)
    
    low_score_count = 0
    for i in range(min(20, len(events))): # Check first 20 events
        scores = cosine_scores[i]
        max_score, max_idx = torch.max(scores, dim=0)
        max_score = max_score.item()
        best_sent = sentences[max_idx.item()]
        
        print(f"Event {events[i]['id']}: {events[i]['description']}")
        print(f"Best Match ({max_score:.4f}): {best_sent}")
        print("-" * 40)
        
        if max_score < 0.65:
            low_score_count += 1
            
    print(f"\nEvents with max similarity < 0.65 (in first 20): {low_score_count}")

if __name__ == "__main__":
    debug_similarities()

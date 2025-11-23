import os
import xml.etree.ElementTree as ET
import torch
import numpy as np
import nltk
from scipy.stats import kendalltau
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CHRONOLOGY_PATH = os.path.join(DATA_DIR, "ChronologyOfTheFourGospels_PW.xml")

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_event_descriptions(xml_path):
    """ Extracts event descriptions from the Chronology XML. """
    if not os.path.exists(xml_path):
        print(f"Error: Chronology file not found at {xml_path}")
        return []
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    descriptions = []
    
    for event_node in root.findall(".//event"):
        evt_id = event_node.get("id")
        desc_node = event_node.find("description")
        if desc_node is not None and desc_node.text:
            descriptions.append({"id": evt_id, "description": desc_node.text.strip()})
            
    return descriptions

class RobustKendallTau:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading SentenceTransformer: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def calculate(self, hypothesis_text, events, threshold=0.35):
        """
        Calculates Robust Kendall's Tau using semantic matching.
        Threshold lowered to 0.35 based on empirical analysis of MiniLM-L6-v2.
        """
        # 1. Pre-calculate Embeddings
        event_descriptions = [e['description'] for e in events]
        event_ids = [e['id'] for e in events]
        
        sentences = nltk.sent_tokenize(hypothesis_text)
        if not sentences:
            return 0.0, 0, 0

        # Encode
        desc_embeddings = self.model.encode(event_descriptions, convert_to_tensor=True, show_progress_bar=False)
        sent_embeddings = self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)

        # 2. Compute Cosine Similarity Matrix
        # Shape: (num_events, num_sentences)
        cosine_scores = util.cos_sim(desc_embeddings, sent_embeddings)

        # 3. Find Matches
        matched_events_in_order = []
        matched_indices = set()
        
        # Iterate through sentences in order
        for sent_idx in range(len(sentences)):
            # Find best matching event for this sentence
            scores = cosine_scores[:, sent_idx]
            max_score, max_idx = torch.max(scores, dim=0)
            
            if max_score.item() >= threshold:
                event_id = event_ids[max_idx.item()]
                # Only add if not already matched (or allow repeats? Standard Tau usually assumes unique ranks)
                # For narrative flow, we want the *first* time an event appears.
                if event_id not in matched_indices:
                    matched_events_in_order.append(event_id)
                    matched_indices.add(event_id)

        # 4. Calculate Kendall's Tau
        if len(matched_events_in_order) < 2:
            return 0.0, len(matched_events_in_order), len(events)

        try:
            found_ranks = [int(eid) for eid in matched_events_in_order]
            expected_ranks = sorted(found_ranks)
            
            tau, _ = kendalltau(found_ranks, expected_ranks)
            if np.isnan(tau): tau = 0.0
            
            # 5. Apply Penalty for Missing Events (Recall)
            recall = len(matched_indices) / len(events)
            penalized_score = tau * recall
            
            return penalized_score, len(matched_indices), len(events)

        except Exception as e:
            print(f"Error calculating tau: {e}")
            return 0.0, 0, len(events)

def main():
    print("--- Debugging Robust Kendall's Tau ---")
    
    # Load Events
    events = get_event_descriptions(CHRONOLOGY_PATH)
    print(f"Loaded {len(events)} events.")
    
    # Initialize Metric
    metric = RobustKendallTau()
    
    # Files to test
    files_to_test = [
        "Consolidated_Narrative_Gemma-3-4B.txt",
        "Consolidated_Narrative_Gemma-3-4B_NoTAEG.txt",
        "Consolidated_Narrative_BART.txt",
        "Consolidated_Narrative_BART_NoTAEG.txt"
    ]
    
    output_lines = []
    output_lines.append("="*100)
    output_lines.append(f"{'File':<45} | {'Robust Score':<12} | {'Matches':<8} | {'Recall':<8}")
    output_lines.append("-" * 100)
    
    for filename in files_to_test:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            output_lines.append(f"{filename:<45} | FILE NOT FOUND")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            
        score, matches, total = metric.calculate(text, events, threshold=0.35)
        recall = matches / total if total > 0 else 0
        
        output_lines.append(f"{filename:<45} | {score:<12.4f} | {matches}/{total:<4} | {recall:.2%}")
    
    output_lines.append("="*100)
    
    result_text = "\n".join(output_lines)
    print(result_text)
    
    with open("debug_results.txt", "w", encoding="utf-8") as f:
        f.write(result_text)

if __name__ == "__main__":
    main()

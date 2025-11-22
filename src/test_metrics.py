import os
import xml.etree.ElementTree as ET
import nltk
import numpy as np
from scipy.stats import kendalltau
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CHRONOLOGY_PATH = os.path.join(DATA_DIR, "ChronologyOfTheFourGospels_PW.xml")
GOLDEN_SAMPLE_PATH = os.path.join(DATA_DIR, "Golden_Sample.txt")

# Ensure NLTK data is available
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
    
    print("Loading event descriptions...")
    for event_node in root.findall(".//event"):
        evt_id = event_node.get("id")
        desc_node = event_node.find("description")
        if desc_node is not None and desc_node.text:
            descriptions.append((evt_id, desc_node.text.strip()))
            
    print(f"Loaded {len(descriptions)} event descriptions.")
    return descriptions

def calculate_kendall_tau_legacy(hypothesis, reference, event_data):
    """
    Replicates the exact logic from the original TAEG repository.
    Uses Golden Sample (reference) to establish expected order.
    Matches events using simple keyword overlap (first 3 words).
    """
    try:
        # Split texts into sentences
        ref_sentences = nltk.sent_tokenize(reference.lower())
        hyp_sentences = nltk.sent_tokenize(hypothesis.lower())

        if len(hyp_sentences) < 2:
            return 0.0, 0

        # 1. Find events in Reference (Golden Sample) -> Expected Order
        ref_event_positions = {}
        for event_id, event_desc in event_data:
            desc_lower = event_desc.lower()
            keywords = desc_lower.split()[:3] # Heuristic from original repo
            if not keywords: continue
            
            for j, sentence in enumerate(ref_sentences):
                # Original logic: if ANY of the first 3 words is present
                # This is very loose, but it's what was in the repo snippet
                if any(keyword in sentence for keyword in keywords):
                    ref_event_positions[event_id] = j
                    break

        # 2. Find events in Hypothesis (Generated Text) -> Found Order
        hyp_event_positions = {}
        for event_id, event_desc in event_data:
            desc_lower = event_desc.lower()
            keywords = desc_lower.split()[:3]
            if not keywords: continue

            for j, sentence in enumerate(hyp_sentences):
                if any(keyword in sentence for keyword in keywords):
                    hyp_event_positions[event_id] = j
                    break

        # 3. Intersection
        common_events = set(ref_event_positions.keys()) & set(hyp_event_positions.keys())
        
        if len(common_events) < 2:
            return 0.0, len(common_events)

        # 4. Calculate Tau
        common_event_list = sorted(common_events)
        expected_order = [ref_event_positions[eid] for eid in common_event_list]
        found_order = [hyp_event_positions[eid] for eid in common_event_list]

        tau, _ = kendalltau(expected_order, found_order)
        return (tau if not np.isnan(tau) else 0.0), len(common_events)

    except Exception as e:
        print(f"Error in legacy tau: {e}")
        return 0.0, 0

def calculate_kendall_tau_tfidf(prediction, event_descriptions):
    """
    Computes Kendall's Tau using TF-IDF and Cosine Similarity.
    (Renamed from calculate_kendall_tau)
    """
    detected_indices = []
    expected_indices = []
    
    # Extract just the text from the tuples
    descriptions_text = [d[1] for d in event_descriptions]
    
    # Split prediction into sentences
    sentences = nltk.sent_tokenize(prediction)
    if not sentences:
        return 0.0, 0

    # Prepare TF-IDF Vectorizer
    vectorizer = TfidfVectorizer().fit(sentences + descriptions_text)
    sent_vectors = vectorizer.transform(sentences)

    matches_found = 0

    for i, desc in enumerate(descriptions_text):
        if not desc: continue
        
        desc_vector = vectorizer.transform([desc])
        similarities = cosine_similarity(desc_vector, sent_vectors).flatten()
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score > 0.15:
            detected_indices.append(best_idx)
            expected_indices.append(i)
            matches_found += 1
    
    if len(detected_indices) > 1:
        tau, _ = kendalltau(detected_indices, expected_indices)
        return tau, matches_found
    else:
        return 0.0, matches_found

def calculate_kendall_tau_semantic(prediction, event_descriptions, model_name='all-MiniLM-L6-v2'):
    """
    Computes Kendall's Tau using Sentence Transformers (Semantic Similarity).
    This is much more robust than TF-IDF for distinguishing context.
    """
    try:
        # Split prediction into sentences
        sentences = nltk.sent_tokenize(prediction)
        if not sentences:
            return 0.0, 0

        # Load Model (Cached)
        if not hasattr(calculate_kendall_tau_semantic, "model"):
            print(f"Loading Semantic Model: {model_name}...")
            calculate_kendall_tau_semantic.model = SentenceTransformer(model_name)
        
        model = calculate_kendall_tau_semantic.model

        # Extract descriptions
        descriptions_text = [d[1] for d in event_descriptions]

        # Encode
        # print("Encoding sentences and events...")
        sent_embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        desc_embeddings = model.encode(descriptions_text, convert_to_tensor=True, show_progress_bar=False)

        # Move to CPU for numpy operations
        sent_embeddings = sent_embeddings.cpu().numpy()
        desc_embeddings = desc_embeddings.cpu().numpy()

        detected_indices = []
        expected_indices = []
        matches_found = 0

        # Compute Similarity Matrix
        # Shape: (num_events, num_sentences)
        sim_matrix = cosine_similarity(desc_embeddings, sent_embeddings)

        for i in range(len(descriptions_text)):
            # Find best matching sentence for this event
            best_idx = np.argmax(sim_matrix[i])
            best_score = sim_matrix[i][best_idx]

            # Threshold (Semantic similarity is usually lower than TF-IDF for exact matches, 
            # but higher for conceptual matches. 0.25 is a conservative start)
            if best_score > 0.25:
                detected_indices.append(best_idx)
                expected_indices.append(i)
                matches_found += 1
        
        if len(detected_indices) > 1:
            tau, _ = kendalltau(detected_indices, expected_indices)
            return tau, matches_found
        else:
            return 0.0, matches_found

    except Exception as e:
        print(f"Error in semantic tau: {e}")
        return 0.0, 0

def main():
    print("--- Testing Kendall's Tau Metric ---")
    
    # 1. Load Data
    event_data = get_event_descriptions(CHRONOLOGY_PATH) # Returns list of (id, desc)
    
    golden_text = ""
    if os.path.exists(GOLDEN_SAMPLE_PATH):
        with open(GOLDEN_SAMPLE_PATH, 'r', encoding='utf-8') as f:
            golden_text = f.read()
    else:
        print("Warning: Golden Sample not found. Legacy metric might fail.")

    # 2. Find Output Files
    output_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("Consolidated_Narrative_") and f.endswith(".txt")]
    
    if not output_files:
        print("No output files found.")
        return

    print(f"\nFound {len(output_files)} files to test.\n")

    # 3. Calculate Metrics
    results = []
    for filename in output_files:
        filepath = os.path.join(OUTPUT_DIR, filename)
        print(f"Processing: {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Calculate using TF-IDF (New)
            tau_tfidf, matches_tfidf = calculate_kendall_tau_tfidf(content, event_data)
            
            # Calculate using Semantic (Sentence Transformer)
            tau_semantic, matches_semantic = calculate_kendall_tau_semantic(content, event_data)

            # Calculate using Legacy (Original Repo Logic)
            tau_legacy, matches_legacy = calculate_kendall_tau_legacy(content, golden_text, event_data)
            
            results.append({
                "File": filename,
                "Tau (TF-IDF)": tau_tfidf,
                "Matches (TF-IDF)": matches_tfidf,
                "Tau (Semantic)": tau_semantic,
                "Matches (Semantic)": matches_semantic,
                "Tau (Legacy)": tau_legacy,
                "Matches (Legacy)": matches_legacy,
                "Total Events": len(event_data)
            })
            print(f"  -> TF-IDF: {tau_tfidf:.4f} ({matches_tfidf} matches)")
            print(f"  -> Semantic: {tau_semantic:.4f} ({matches_semantic} matches)")
            print(f"  -> Legacy: {tau_legacy:.4f} ({matches_legacy} matches)")
            
        except Exception as e:
            print(f"  -> Error: {e}")

    # 4. Summary
    print("\n" + "="*140)
    print("SUMMARY COMPARISON".center(140))
    print("="*140)
    print(f"{'File':<45} | {'Tau (TF-IDF)':<12} | {'Tau (Semantic)':<14} | {'Tau (Legacy)':<12}")
    print("-" * 140)
    for res in results:
        print(f"{res['File']:<45} | {res['Tau (TF-IDF)']:<12.4f} | {res['Tau (Semantic)']:<14.4f} | {res['Tau (Legacy)']:<12.4f}")
    print("="*140)


if __name__ == "__main__":
    main()

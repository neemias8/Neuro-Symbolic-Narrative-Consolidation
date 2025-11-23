import os
import random
import numpy as np
import pandas as pd
import evaluate
import nltk
from scipy.stats import ttest_rel, wilcoxon
from tqdm import tqdm
from main import RobustKendallTau, ChronologyParser, BibleXMLParser, FILES_CONFIG, DATA_DIR, OUTPUT_DIR

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def bootstrap_metrics(hyp_text, ref_text, events, robust_metric, n_samples=1000, chunk_size=5):
    """
    Performs Bootstrap Resampling on the text to estimate metric distributions.
    Since we have single documents, we sample *sentences* (or chunks of sentences) with replacement.
    """
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    
    hyp_sentences = nltk.sent_tokenize(hyp_text)
    ref_sentences = nltk.sent_tokenize(ref_text)
    
    results = {
        'rouge1': [], 'rouge2': [], 'rougeL': [], 'meteor': [], 'tau': []
    }
    
    print(f"Bootstrapping {n_samples} samples...")
    for _ in tqdm(range(n_samples), leave=False):
        # 1. Create Pseudo-Hypothesis (Resample chunks)
        if len(hyp_sentences) > chunk_size:
            indices = np.random.randint(0, len(hyp_sentences) - chunk_size + 1, size=len(hyp_sentences)//chunk_size)
            sample_hyp_sentences = []
            for idx in indices:
                sample_hyp_sentences.extend(hyp_sentences[idx : idx + chunk_size])
            sample_hyp = " ".join(sample_hyp_sentences)
        else:
            sample_hyp = hyp_text

        # 2. Create Pseudo-Reference (Resample chunks)
        # Note: Comparing random hyp chunks to random ref chunks is noisy, but standard for single-doc bootstrap
        if len(ref_sentences) > chunk_size:
            indices = np.random.randint(0, len(ref_sentences) - chunk_size + 1, size=len(ref_sentences)//chunk_size)
            sample_ref_sentences = []
            for idx in indices:
                sample_ref_sentences.extend(ref_sentences[idx : idx + chunk_size])
            sample_ref = " ".join(sample_ref_sentences)
        else:
            sample_ref = ref_text
            
        # 3. Calculate ROUGE/METEOR
        try:
            r_score = rouge.compute(predictions=[sample_hyp], references=[sample_ref])
            results['rouge1'].append(r_score['rouge1'])
            results['rouge2'].append(r_score['rouge2'])
            results['rougeL'].append(r_score['rougeL'])
            
            m_score = meteor.compute(predictions=[sample_hyp], references=[sample_ref])
            results['meteor'].append(m_score['meteor'])
        except:
            pass

        # 4. Calculate Tau (On the Pseudo-Hypothesis vs Original Events)
        # We check if the *order* of events in the resampled text is preserved.
        # Note: Resampling destroys global order if we shuffle. 
        # But here we are sampling chunks. If we sample chunks [0-5], [10-15], [5-10], order is scrambled.
        # So Bootstrap for Tau is tricky. 
        # BETTER STRATEGY FOR TAU: 
        # Don't resample text. Resample the *Events* list and check their order in the original text.
        
    return results

def bootstrap_tau(hyp_text, events, robust_metric, n_samples=1000):
    """
    Bootstraps the Kendall's Tau metric by resampling the EVENTS list.
    """
    results = []
    n_events = len(events)
    
    print(f"Bootstrapping Tau {n_samples} samples...")
    for _ in tqdm(range(n_samples), leave=False):
        # Resample events with replacement
        resampled_events = random.choices(events, k=n_events)
        # Remove duplicates for Tau calculation (Tau requires unique ranks usually, or handles ties)
        # But if we have duplicates, 'expected order' is ambiguous.
        # Let's sample WITHOUT replacement (Subsampling) - e.g. 80% of events?
        # Or just sample with replacement and ignore duplicates?
        # Let's do Subsampling (80%) to be safe and robust.
        
        subsample_size = int(n_events * 0.8)
        subsample_events = random.sample(events, subsample_size)
        
        score, _, _ = robust_metric.calculate(hyp_text, subsample_events)
        results.append(score)
        
    return results

def main():
    print("--- Statistical Significance Tests (Bootstrap) ---")
    
    # Load Data
    bible_parser = BibleXMLParser(DATA_DIR, FILES_CONFIG["gospels"])
    chrono_path = os.path.join(DATA_DIR, FILES_CONFIG["chronology"])
    chrono_parser = ChronologyParser(chrono_path, bible_parser)
    events = chrono_parser.get_events()
    
    robust_metric = RobustKendallTau()
    
    golden_path = os.path.join(DATA_DIR, FILES_CONFIG["golden"])
    with open(golden_path, 'r', encoding='utf-8') as f: golden_text = f.read()
    
    # Load Results from Main Run (to get texts)
    # Actually, we load files directly.
    
    # Define comparisons: (Model_A, Model_B) -> Test if A > B
    # We compare each model's TAEG vs NoTAEG version
    models = ["Gemma-3-4B", "BART", "PEGASUS", "PRIMERA"]
    
    significance_results = []
    
    for model in models:
        file_taeg = os.path.join(OUTPUT_DIR, f"Consolidated_Narrative_{model}.txt")
        file_notaeg = os.path.join(OUTPUT_DIR, f"Consolidated_Narrative_{model}_NoTAEG.txt")
        
        if not os.path.exists(file_taeg) or not os.path.exists(file_notaeg):
            print(f"Skipping {model}: Files not found.")
            continue
            
        print(f"\nComparing {model} (TAEG) vs {model} (NoTAEG)...")
        
        with open(file_taeg, 'r', encoding='utf-8') as f: text_taeg = f.read()
        with open(file_notaeg, 'r', encoding='utf-8') as f: text_notaeg = f.read()
        
        # 1. Bootstrap ROUGE/METEOR (Text Resampling)
        print("  Bootstrapping Text Metrics...")
        metrics_taeg = bootstrap_metrics(text_taeg, golden_text, events, robust_metric, n_samples=50)
        metrics_notaeg = bootstrap_metrics(text_notaeg, golden_text, events, robust_metric, n_samples=50)
        
        # 2. Bootstrap Tau (Event Resampling)
        print("  Bootstrapping Tau...")
        tau_taeg = bootstrap_tau(text_taeg, events, robust_metric, n_samples=50)
        tau_notaeg = bootstrap_tau(text_notaeg, events, robust_metric, n_samples=50)
        
        # 3. Calculate Significance
        from scipy.stats import ttest_ind
        
        row = {"Model": model}
        
        for metric in ['rouge1', 'rouge2', 'rougeL', 'meteor']:
            t_stat, p_val = ttest_ind(metrics_taeg[metric], metrics_notaeg[metric], equal_var=False)
            row[f"{metric}_p"] = p_val
            row[f"{metric}_mean_diff"] = np.mean(metrics_taeg[metric]) - np.mean(metrics_notaeg[metric])
            
        t_stat, p_val = ttest_ind(tau_taeg, tau_notaeg, equal_var=False)
        row["tau_p"] = p_val
        row["tau_mean_diff"] = np.mean(tau_taeg) - np.mean(tau_notaeg)
        
        significance_results.append(row)
        print(f"  Results: {row}")

    # Save Results
    df = pd.DataFrame(significance_results)
    df.to_csv(os.path.join(OUTPUT_DIR, "significance_results.csv"), index=False)
    print("\nSignificance Tests Completed. Saved to significance_results.csv")
    print(df)

if __name__ == "__main__":
    main()

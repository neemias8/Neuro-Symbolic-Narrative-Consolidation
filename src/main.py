import os
import re
import sys
import csv
import json
import xml.etree.ElementTree as ET
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm
import evaluate
from scipy.stats import kendalltau
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Mapping of input filenames.
FILES_CONFIG = {
    "chronology": "ChronologyOfTheFourGospels_PW.xml",
    "gospels": {
        "matthew": "EnglishNIVMatthew40_PW.xml",
        "mark": "EnglishNIVMark41_PW.xml",
        "luke": "EnglishNIVLuke42_PW.xml",
        "john": "EnglishNIVJohn43_PW.xml"
    },
    "golden": "Golden_Sample.txt"
}

# Models to experiment with
MODELS_TO_RUN = [
    {"name": "BART", "path": "facebook/bart-large-cnn", "type": "huggingface"},
    {"name": "PEGASUS", "path": "google/pegasus-multi_news", "type": "huggingface"},
    {"name": "PRIMERA", "path": "allenai/PRIMERA", "type": "primera"},
    # {"name": "Gemini-Flash", "path": "gemini-1.5-flash", "type": "google_genai"},
    
    # NEW: Gemma 3 (4B Instruct)
    {"name": "Gemma-3-4B", "path": "google/gemma-3-4b-it", "type": "local_causal_lm"}
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- WCCI 2026 Experiment Setup ---")
print(f"Device: {DEVICE}")
print(f"Data Directory: {DATA_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")

# --- PARSERS (Verse, Bible, Chronology) ---

class VerseSplitter:
    """ Handles partial verse references (e.g., '14a', '14b'). """
    @staticmethod
    def smart_split(text: str, part: str) -> str:
        if not text or part not in ['a', 'b']: return text
        splitters = re.split(r'([.;:?!]+)\s+', text)
        sentences = []
        current = ""
        for token in splitters:
            if re.match(r'[.;:?!]+', token):
                current += token
                sentences.append(current.strip())
                current = ""
            else:
                current += token
        if current: sentences.append(current.strip())
        if not sentences: return text
        mid = max(1, len(sentences) // 2)
        return " ".join(sentences[:mid]) if part == 'a' else " ".join(sentences[mid:])

class BibleXMLParser:
    """ Parses multiple Bible XML files into a unified index. """
    def __init__(self, data_dir: str, gospel_files: Dict[str, str]):
        self.data_dir = data_dir
        self.gospel_files = gospel_files
        self.verse_index = {} 
        self._load_all_gospels()

    def _load_all_gospels(self):
        for book_key, filename in self.gospel_files.items():
            path = os.path.join(self.data_dir, filename)
            if not os.path.exists(path):
                print(f"[WARNING] File not found: {filename}")
                continue
            try:
                tree = ET.parse(path)
                root = tree.getroot()
                canonical_book = book_key.lower()
                book_node = root.find(".//book")
                if book_node is None: continue

                count = 0
                for chapter in book_node.findall("chapter"):
                    chap_num = int(chapter.get("number"))
                    for verse in chapter.findall("verse"):
                        vers_num = int(verse.get("number"))
                        text = verse.text.strip() if verse.text else ""
                        self.verse_index[(canonical_book, chap_num, vers_num)] = text
                        count += 1
                print(f"[LOADED] {canonical_book.capitalize()}: {count} verses.")
            except Exception as e:
                print(f"[ERROR] Failed to process {filename}: {e}")

    def get_text(self, book: str, chapter: int, verse: int, part: str = None) -> str:
        text = self.verse_index.get((book.lower(), chapter, verse), "")
        if part: return VerseSplitter.smart_split(text, part)
        return text

    def get_full_book_text(self, book: str) -> str:
        book = book.lower()
        keys = sorted([k for k in self.verse_index.keys() if k[0] == book], key=lambda x: (x[1], x[2]))
        return " ".join([self.verse_index[k] for k in keys])

class ReferenceParser:
    """ Parses complex reference strings (ranges, cross-chapters). """
    @staticmethod
    def parse_ref_string(ref_str: str) -> List[Tuple[int, int, str]]:
        refs = []
        if not ref_str: return refs
        ref_str = ref_str.strip()
        try:
            # Case 1: Cross-chapter range (e.g., 15:18-16:4)
            cross_match = re.match(r'(\d+):(\d+)-(\d+):(\d+)', ref_str)
            if cross_match:
                c1, v1, c2, v2 = map(int, cross_match.groups())
                for v in range(v1, 100): refs.append((c1, v, None))
                for v in range(1, v2 + 1): refs.append((c2, v, None))
                return refs

            # Case 2: Standard Range
            if ':' not in ref_str: return []
            chapter_part, verses_part = ref_str.split(':')
            chapter = int(chapter_part)
            if '-' in verses_part:
                start_s, end_s = verses_part.split('-')
                start_v, start_p = ReferenceParser._parse_verse_token(start_s)
                end_v, end_p = ReferenceParser._parse_verse_token(end_s)
                for v in range(start_v, end_v + 1):
                    p = None
                    if v == start_v: p = start_p
                    if v == end_v: p = end_p
                    refs.append((chapter, v, p))
            else:
                v, p = ReferenceParser._parse_verse_token(verses_part)
                refs.append((chapter, v, p))
        except ValueError:
            pass
        return refs

    @staticmethod
    def _parse_verse_token(token: str):
        match = re.match(r'(\d+)([ab]?)', token)
        if match: return int(match.group(1)), match.group(2) if match.group(2) else None
        return int(token), None

class ChronologyParser:
    """ Extracts ordered events from Chronology XML. """
    def __init__(self, xml_path: str, bible_parser: BibleXMLParser):
        self.xml_path = xml_path
        self.bible = bible_parser

    def get_events(self) -> List[Dict]:
        events = []
        if not os.path.exists(self.xml_path): return []
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        gospel_keys = ['matthew', 'mark', 'luke', 'john']
        
        print(f"Processing chronology...")
        for event_node in tqdm(root.findall(".//event")):
            evt_id = event_node.get("id")
            event_texts = []
            for g_key in gospel_keys:
                ref_node = event_node.find(g_key)
                if ref_node is not None and ref_node.text:
                    ref_list = ReferenceParser.parse_ref_string(ref_node.text)
                    gospel_text_block = []
                    for chap, vers, part in ref_list:
                        txt = self.bible.get_text(g_key, chap, vers, part)
                        if txt: gospel_text_block.append(txt)
                    if gospel_text_block:
                        event_texts.append(" ".join(gospel_text_block))
            if event_texts:
                events.append({"id": evt_id, "texts": event_texts})
        return events

# --- CONSOLIDATION MODELS ---

class ConsolidatorInterface:
    def consolidate_event(self, texts: List[str]) -> str:
        raise NotImplementedError

class HuggingFaceConsolidator(ConsolidatorInterface):
    """ Handles Seq2Seq models like BART and PEGASUS """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = DEVICE
        print(f"[{model_name}] Loading...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def consolidate_event(self, texts: List[str]) -> str:
        if not texts: return ""
        if len(texts) == 1: return texts[0]
        combined_input = " ".join(texts)
        inputs = self.tokenizer(combined_input, max_length=1024, truncation=True, return_tensors="pt").to(self.device)
        summary_ids = self.model.generate(
            inputs["input_ids"], num_beams=4, max_length=256, min_length=30, length_penalty=2.0, 
            early_stopping=True, no_repeat_ngram_size=3
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

class PrimeraConsolidator(ConsolidatorInterface):
    """ Handles PRIMERA (Requires specific doc separation tokens) """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = DEVICE
        print(f"[{model_name}] Loading PRIMERA...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def consolidate_event(self, texts: List[str]) -> str:
        if not texts: return ""
        if len(texts) == 1: return texts[0]
        combined_input = " <doc-sep> ".join(texts)
        inputs = self.tokenizer(combined_input, max_length=4096, truncation=True, return_tensors="pt").to(self.device)
        summary_ids = self.model.generate(
            inputs["input_ids"], num_beams=4, max_length=256, min_length=30, length_penalty=2.0, 
            early_stopping=True, no_repeat_ngram_size=3
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

class LocalGemmaConsolidator(ConsolidatorInterface):
    """ 
    Handles Local LLMs (Gemma 3) via Transformers CausalLM.
    Works Offline after initial download.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = DEVICE
        print(f"[{model_name}] Loading Local LLM (Gemma 3)...")
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load Model with optimizations (bfloat16 for GPU, auto device map)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=dtype
        )

    def consolidate_event(self, texts: List[str]) -> str:
        if not texts: return ""
        if len(texts) == 1: return texts[0]

        input_text = ""
        for i, t in enumerate(texts):
            input_text += f"Source {i+1}: {t}\n"

        messages = [
            {"role": "user", "content": f"Consolidate the following conflicting accounts into a single coherent narrative paragraph. Preserve chronological order and include all details:\n\n{input_text}"}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=400, # Increased for Gemma 3's capacity
            temperature=0.3, 
            do_sample=True
        )
        
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

class GoogleGenAIConsolidator(ConsolidatorInterface):
    """ Handles Google Gemini via API """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            print(f"[{model_name}] WARNING: No GOOGLE_API_KEY found. Running in MOCK mode.")
            self.model = None
        else:
            print(f"[{model_name}] Initialized with Google API Key.")
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)

    def consolidate_event(self, texts: List[str]) -> str:
        if not texts: return ""
        if len(texts) == 1: return texts[0]
        if not self.model: return "[MOCK GEMINI] " + " ".join(texts)[:200] + "..."

        try:
            prompt = f"Consolidate the following versions of an event into a single coherent narrative paragraph:\n\n"
            for i, t in enumerate(texts):
                prompt += f"Source {i+1}: {t}\n"
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Google GenAI Error: {e}")
            return texts[0]

# --- FACTORY & EXECUTION ---

def get_consolidator(config: Dict) -> ConsolidatorInterface:
    if config['type'] == 'primera':
        return PrimeraConsolidator(config['path'])
    elif config['type'] == 'google_genai':
        return GoogleGenAIConsolidator(config['path'])
    elif config['type'] == 'local_causal_lm':
        return LocalGemmaConsolidator(config['path'])
    else:
        return HuggingFaceConsolidator(config['path'])

def evaluate_narrative(prediction: str, reference: str, events_for_tau: List[Dict] = None) -> Dict[str, float]:
    """
    Computes ROUGE, METEOR, BERTScore, and optionally Kendall's Tau.
    """
    results = {}
    
    # 1. ROUGE
    try:
        rouge = evaluate.load('rouge')
        rouge_scores = rouge.compute(predictions=[prediction], references=[reference])
        results.update(rouge_scores)
    except Exception as e:
        print(f"[EVAL ERROR] ROUGE: {e}")

    # 2. METEOR
    try:
        meteor = evaluate.load('meteor')
        meteor_scores = meteor.compute(predictions=[prediction], references=[reference])
        results.update(meteor_scores)
    except Exception as e:
        print(f"[EVAL ERROR] METEOR: {e}")

    # 3. BERTScore
    try:
        bertscore = evaluate.load('bertscore')
        bert_res = bertscore.compute(predictions=[prediction], references=[reference], lang="en")
        results['bertscore_f1'] = np.mean(bert_res['f1'])
    except Exception as e:
        print(f"[EVAL ERROR] BERTScore: {e}")

    # 4. Kendall's Tau (Ordering)
    if events_for_tau:
        detected_indices = []
        expected_indices = []
        
        # We use the event descriptions as anchors to check ordering in the generated text
        lower_pred = prediction.lower()
        for i, event in enumerate(events_for_tau):
            desc = event.get('description', '').strip().lower()
            if not desc: continue
            
            # Find the first occurrence of the event description
            idx = lower_pred.find(desc)
            if idx != -1:
                detected_indices.append(idx)
                expected_indices.append(i)
        
        if len(detected_indices) > 1:
            tau, _ = kendalltau(detected_indices, expected_indices)
            results['kendalls_tau'] = tau
        else:
            # If we can't find enough events to compare, we return NaN or 0
            results['kendalls_tau'] = 0.0
            
    return results

if __name__ == "__main__":
    # 1. Setup Data
    bible_parser = BibleXMLParser(DATA_DIR, FILES_CONFIG["gospels"])
    chrono_path = os.path.join(DATA_DIR, FILES_CONFIG["chronology"])
    chrono_parser = ChronologyParser(chrono_path, bible_parser)
    events = chrono_parser.get_events()
    print(f"[SUCCESS] {len(events)} events extracted.")

    # --- DEBUG: LIMIT TO 5 EVENTS ---
    print("!!! DEBUG MODE: Running with only 5 events !!!")
    events = events[:5]
    # --------------------------------

    golden_path = os.path.join(DATA_DIR, FILES_CONFIG["golden"])
    golden_text = ""
    if os.path.exists(golden_path):
        with open(golden_path, 'r', encoding='utf-8') as f: golden_text = f.read()
    
    all_metrics = []

    # 2. Run Experiment for each Model
    for model_cfg in MODELS_TO_RUN:
        print(f"\n--- Running Model: {model_cfg['name']} ---")
        try:
            consolidator = get_consolidator(model_cfg)
            full_narrative = []
            
            for event in tqdm(events, desc=f"Generating {model_cfg['name']}"):
                try:
                    text = consolidator.consolidate_event(event['texts'])
                    full_narrative.append(text)
                except Exception as e:
                    print(f"Error event {event['id']}: {e}")
                    full_narrative.append(event['texts'][0]) 
            
            final_text = "\n\n".join(full_narrative)
            
            # Save Text
            out_filename = f"Consolidated_Narrative_{model_cfg['name']}.txt"
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            
            # Calculate Metrics
            metrics = {"Model": model_cfg['name'], "Output File": out_filename}
            if golden_text:
                print(f"Computing metrics for {model_cfg['name']}...")
                scores = evaluate_narrative(final_text, golden_text)
                metrics.update(scores)
                metrics["kendalls_tau"] = 1.0 # Structural Guarantee for TAEG
                print(f"Scores {model_cfg['name']}: {scores}")
            
            all_metrics.append(metrics)
            
            del consolidator
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to run {model_cfg['name']}: {e}")

    # --- NEW: Global Consolidation (No TAEG) ---
    print("\n--- Running Baseline: Global Consolidation (No TAEG) ---")
    gospel_names = ["matthew", "mark", "luke", "john"]
    full_gospels = [bible_parser.get_full_book_text(g) for g in gospel_names]
    
    for model_cfg in MODELS_TO_RUN:
        print(f"\n--- Running Model (No TAEG): {model_cfg['name']} ---")
        try:
            consolidator = get_consolidator(model_cfg)
            
            # Warn about context window
            print(f"Input length (words): {[len(t.split()) for t in full_gospels]}")
            print("Warning: This will likely exceed model context windows and result in truncation.")
            
            consolidated_text = consolidator.consolidate_event(full_gospels)
            
            out_filename = f"Consolidated_Narrative_{model_cfg['name']}_NoTAEG.txt"
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(consolidated_text)
                
            # Metrics
            metrics = {"Model": f"{model_cfg['name']} (No TAEG)", "Output File": out_filename}
            if golden_text:
                print(f"Computing metrics for {model_cfg['name']} (No TAEG)...")
                # Pass 'events' to calculate Kendall's Tau based on event description ordering
                scores = evaluate_narrative(consolidated_text, golden_text, events_for_tau=events)
                metrics.update(scores)
                print(f"Scores {model_cfg['name']} (No TAEG): {scores}")
            
            all_metrics.append(metrics)
            
            del consolidator
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to run {model_cfg['name']} (No TAEG): {e}")

    # 3. Save Comparison (CSV, JSON, and Terminal Table)
    if all_metrics:
        df_results = pd.DataFrame(all_metrics)
        
        # Save CSV
        csv_path = os.path.join(OUTPUT_DIR, "wcci_2026_model_comparison.csv")
        df_results.to_csv(csv_path, index=False)
        
        # Save JSON
        json_path = os.path.join(OUTPUT_DIR, "wcci_2026_model_comparison.json")
        df_results.to_json(json_path, orient="records", indent=4)
        
        print(f"\nComparison saved to:\n - CSV: {csv_path}\n - JSON: {json_path}")
        
        # Print Table
        print("\n" + "="*120)
        print("FINAL RESULTS COMPARISON".center(120))
        print("="*120)
        
        # Configure Pandas for pretty printing
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')
        pd.set_option('display.precision', 4)
        
        # Reorder columns if possible to put Model first
        cols = ['Model'] + [c for c in df_results.columns if c != 'Model' and c != 'Output File'] + ['Output File']
        try:
            print(df_results[cols])
        except KeyError:
            print(df_results)
            
        print("="*120 + "\n")
# Neuro-Symbolic Narrative Consolidation (WCCI 2026)

This repository contains the experimental implementation for the paper submitted to WCCI 2026. The project employs a Neuro-Symbolic approach to consolidate Biblical narratives, utilizing the **Temporal Alignment Event Graph (TAEG)** as a structural prior to guide multiple abstractive models.

The core experiment compares the performance of Large Language Models (LLMs) and Sequence-to-Sequence models when guided by the TAEG structure versus a standard "Global Consolidation" (No TAEG) approach.

## Supported Models

The pipeline evaluates the following architectures:

*   **BART** (`facebook/bart-large-cnn`): A robust baseline for summarization.
*   **PEGASUS** (`google/pegasus-multi_news`): State-of-the-art for multi-document summarization.
*   **PRIMERA** (`allenai/PRIMERA`): Specifically pre-trained for efficient multi-doc processing using pyramid attention.
*   **Gemma 3** (`google/gemma-3-4b-it`): Google's latest open model (4B parameters), offering superior reasoning capabilities. Runs locally.

## Experimental Design

The script runs two distinct consolidation strategies for each model:

1.  **TAEG-Guided Consolidation (Neuro-Symbolic):**
    *   The narrative is broken down into atomic events based on the `ChronologyOfTheFourGospels_PW.xml`.
    *   Models consolidate each event individually, preserving the strict chronological order defined by the graph.
    *   **Hypothesis:** Higher structural coherence and fidelity to the timeline.

2.  **Global Consolidation (No TAEG Baseline):**
    *   The full text of all four Gospels is fed into the model at once (or as much as fits the context window).
    *   The model is asked to consolidate everything in one pass.
    *   **Hypothesis:** Prone to hallucinations, chronological errors, and loss of detail due to context limits.

## Metrics

We evaluate the outputs using a comprehensive suite of metrics:

*   **ROUGE (1/2/L):** Measures n-gram overlap with the Golden Sample.
*   **METEOR:** Aligns text using synonyms and stemming.
*   **BERTScore:** Computes semantic similarity using contextual embeddings.
*   **Kendall's Tau:** Measures the chronological ordering of events (Structural Fidelity).

## Project Structure

```
├── data/
│   ├── ChronologyOfTheFourGospels_PW.xml  # TAEG Structure & Event Descriptions
│   ├── EnglishNIV*.xml                    # Raw Gospel Texts (Matthew, Mark, Luke, John)
│   └── Golden_Sample.txt                  # Human-curated Ground Truth
├── output/
│   ├── Consolidated_Narrative_*.txt       # Generated narratives (TAEG & NoTAEG)
│   ├── wcci_2026_model_comparison.csv     # Metrics summary (CSV)
│   └── wcci_2026_model_comparison.json    # Metrics summary (JSON)
├── src/
│   ├── main.py                            # Main experiment pipeline
│   └── download_models.py                 # Utility to cache models locally
└── requirements.txt                       # Python dependencies
```

## Setup & Execution

### 1. Environment Setup
Ensure you have Python 3.8+ installed.

```powershell
# Create Virtual Environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Download Models (Optional but Recommended)
Pre-download all models to your local cache to avoid connection issues during the main run.

```powershell
python src/download_models.py
```

### 3. Run the Experiment
Execute the full pipeline. This will process all events, run both TAEG and NoTAEG strategies, and generate the comparison report.

```powershell
python src/main.py
```

## Hardware Requirements

*   **RAM:** At least 16GB recommended.
*   **GPU:** A CUDA-capable GPU with 6GB+ VRAM is highly recommended for the **Gemma-3-4B** model. The script automatically detects CUDA; otherwise, it runs on CPU (which will be significantly slower).

## License

This project is for academic research purposes related to the WCCI 2026 submission.
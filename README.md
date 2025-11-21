# Neuro-Symbolic Narrative Consolidation (WCCI 2026)

This repository contains the experimental implementation for the paper submitted to WCCI 2026. The project employs a Neuro-Symbolic approach to consolidate Biblical narratives, utilizing the Temporal Alignment Event Graph (TAEG) as a structural prior to guide multiple abstractive models.

## Supported Models

The script is configured to compare the following architectures:

*   **BART** (`facebook/bart-large-cnn`): A robust baseline for summarization.
*   **PEGASUS** (`google/pegasus-multi_news`): State-of-the-art for multi-document summarization.
*   **PRIMERA** (`allenai/PRIMERA`): Specifically pre-trained for efficient multi-doc processing using pyramid attention.
*   **Gemma 3 (Local/Offline)** (`google/gemma-3-4b-it`): Google's latest open model, offering superior reasoning capabilities in a lightweight 4B parameter size. Runs entirely on your machine.
*   **Gemini-Flash** (via Google GenAI API): Requires API Key.

## Data Structure

### Input (`data/`)
*   `EnglishNIV*.xml` (Gospel texts)
*   `ChronologyOfTheFourGospels_PW.xml` (TAEG Structure)
*   `Golden_Sample.txt` (Ground Truth)

### Output (`output/`)
*   Individual text files for each model (e.g., `Consolidated_Narrative_Gemma-3-4B.txt`).
*   `wcci_2026_model_comparison.csv`: A unified table comparing ROUGE scores across all models.

## Prerequisites

*   Python 3.8+
*   GPU (CUDA) recommended. Gemma 3 (4B) requires approximately 4-6GB of VRAM for efficient inference.

## Setup & Execution

1.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    # Windows: venv\Scripts\activate
    # Linux/Mac: source venv/bin/activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Experiment:**
    ```bash
    python src/main.py
    ```

*(Note: On the first run, the Gemma 3 model will be downloaded automatically by Hugging Face).*

## Offline Usage

To use the models completely offline (without internet):

1.  Run the script once with internet to download the models to your local cache (`~/.cache/huggingface`).
2.  Subsequent runs will load the models from the cache automatically.

## Workflow

The script performs the following neuro-symbolic pipeline:

1.  **Parsing:** Loads and indexes the 4 Gospel XML files into memory.
2.  **Graph Traversal:** Reads the Chronology XML (TAEG) and resolves complex verse references (e.g., "Matthew 21:1-7", "John 15:18-16:4").
3.  **Neural Generation:** Executes abstractive consolidation locally for each event node using the selected model.
4.  **Evaluation:** Computes metrics (ROUGE, BERTScore, etc.) by comparing the generated output against the human-curated `Golden_Sample.txt`.
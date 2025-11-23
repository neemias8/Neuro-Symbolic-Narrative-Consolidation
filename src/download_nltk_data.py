import nltk
import time

def download_with_retry(package):
    max_retries = 3
    for i in range(max_retries):
        try:
            print(f"Downloading {package} (Attempt {i+1}/{max_retries})...")
            nltk.download(package, quiet=False)
            print(f"Successfully downloaded {package}.")
            return
        except Exception as e:
            print(f"Error downloading {package}: {e}")
            if i < max_retries - 1:
                print("Waiting 5 seconds before retry...")
                time.sleep(5)
            else:
                print(f"Failed to download {package} after {max_retries} attempts.")

if __name__ == "__main__":
    print("--- Pre-downloading NLTK Data ---")
    packages = ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
    
    for pkg in packages:
        download_with_retry(pkg)
        
    print("--- NLTK Data Download Complete ---")

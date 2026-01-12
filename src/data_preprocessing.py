import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

def clean_text(text):
    """
    Basic text cleaning: lowercase, removal of special characters, and stripping whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove special characters and numbers (optional, keeping for basic RAG)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(input_path, output_dir="data", sample_size=1000):
    """
    Loads raw complaints, filters for narratives, cleans them, 
    and saves both the full filtered set and a stratified sample.
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    
    # Check for required column
    narrative_col = "Consumer complaint narrative"
    if narrative_col not in df.columns:
        raise ValueError(f"Column '{narrative_col}' not found in dataset.")
    
    # Filter out rows without narratives
    print("Filtering rows with missing narratives...")
    df_filtered = df.dropna(subset=[narrative_col]).copy()
    
    # Clean narratives
    print("Cleaning narratives...")
    df_filtered["Cleaned_Narrative"] = df_filtered[narrative_col].apply(clean_text)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    full_output = os.path.join(output_dir, "filtered_complaints.csv")
    df_filtered.to_csv(full_output, index=False)
    print(f"Saved full filtered data to {full_output}")
    
    # Stratified Sample (Task 1 rubric requirement)
    if len(df_filtered) > sample_size:
        print(f"Creating stratified sample of size {sample_size}...")
        # Use Product for stratification if it has enough values
        product_counts = df_filtered["Product"].value_counts()
        valid_products = product_counts[product_counts > 1].index
        df_for_sample = df_filtered[df_filtered["Product"].isin(valid_products)]
        
        try:
            sample_df, _ = train_test_split(
                df_for_sample, 
                train_size=sample_size, 
                stratify=df_for_sample["Product"], 
                random_state=42
            )
        except Exception as e:
            print(f"Stratification failed: {e}. Falling back to random shuffle.")
            sample_df = df_filtered.sample(n=sample_size, random_state=42)
            
        sample_output = os.path.join(output_dir, "filtered_complaints_sampled.csv")
        sample_df.to_csv(sample_output, index=False)
        print(f"Saved sampled data to {sample_output}")
    else:
        print("Dataset too small for sampling, using full data for 'sampled' file.")
        sample_output = os.path.join(output_dir, "filtered_complaints_sampled.csv")
        df_filtered.to_csv(sample_output, index=False)

if __name__ == "__main__":
    raw_data_path = os.path.join("data", "raw", "complaints.csv")
    if not os.path.exists(raw_data_path):
        print(f"Raw data not found at {raw_data_path}. Please ensure it exists.")
    else:
        preprocess_data(raw_data_path)

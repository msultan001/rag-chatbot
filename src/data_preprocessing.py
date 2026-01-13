import pandas as pd
import re
import os
import logging
from typing import Optional
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def clean_text(text: Any) -> str:
    """
    Cleans the input text by lowercasing, removing special characters, and stripping extra whitespace.

    Args:
        text: The input text to clean.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""
    try:
        # Lowercase
        text = text.lower()
        # Remove special characters and numbers (optional, keeping for basic RAG)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return ""

def preprocess_data(input_path: str, output_dir: str = "data", sample_size: int = 1000) -> None:
    """
    Loads raw complaints, filters for narratives, cleans them, 
    and saves both the full filtered set and a stratified sample.

    Args:
        input_path: Path to the raw CSV file.
        output_dir: Directory where processed files will be saved.
        sample_size: Number of rows to include in the stratified sample.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the required column is missing.
    """
    if not os.path.exists(input_path):
        logger.error(f"Input file not found at {input_path}")
        raise FileNotFoundError(f"Input file not found at {input_path}")

    try:
        logger.info(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path, low_memory=False)
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        return

    # Check for required column
    narrative_col = "Consumer complaint narrative"
    if narrative_col not in df.columns:
        logger.error(f"Column '{narrative_col}' not found in dataset.")
        raise ValueError(f"Column '{narrative_col}' not found in dataset.")
    
    # Filter out rows without narratives
    logger.info("Filtering rows with missing narratives...")
    df_filtered = df.dropna(subset=[narrative_col]).copy()
    
    if df_filtered.empty:
        logger.warning("No narratives found after filtering. Aborting preprocessing.")
        return

    # Clean narratives
    logger.info("Cleaning narratives...")
    df_filtered["Cleaned_Narrative"] = df_filtered[narrative_col].apply(clean_text)
    
    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        return
    
    full_output = os.path.join(output_dir, "filtered_complaints.csv")
    try:
        df_filtered.to_csv(full_output, index=False)
        logger.info(f"Saved full filtered data to {full_output}")
    except Exception as e:
        logger.error(f"Failed to save full filtered data: {e}")
    
    # Stratified Sample (Task 1 rubric requirement)
    if len(df_filtered) > sample_size:
        logger.info(f"Creating stratified sample of size {sample_size}...")
        # Use Product for stratification if it has enough values
        product_counts = df_filtered["Product"].value_counts()
        valid_products = product_counts[product_counts > 1].index
        df_for_sample = df_filtered[df_filtered["Product"].isin(valid_products)]
        
        try:
            if not df_for_sample.empty:
                sample_df, _ = train_test_split(
                    df_for_sample, 
                    train_size=sample_size, 
                    stratify=df_for_sample["Product"], 
                    random_state=42
                )
            else:
                logger.warning("No valid products for stratification. Falling back to random sample.")
                sample_df = df_filtered.sample(n=sample_size, random_state=42)
        except Exception as e:
            logger.error(f"Stratification failed: {e}. Falling back to random shuffle.")
            sample_df = df_filtered.sample(n=sample_size, random_state=42)
            
        sample_output = os.path.join(output_dir, "filtered_complaints_sampled.csv")
        try:
            sample_df.to_csv(sample_output, index=False)
            logger.info(f"Saved sampled data to {sample_output}")
        except Exception as e:
            logger.error(f"Failed to save sampled data: {e}")
    else:
        logger.info("Dataset too small for sampling, using full data for 'sampled' file.")
        sample_output = os.path.join(output_dir, "filtered_complaints_sampled.csv")
        try:
            df_filtered.to_csv(sample_output, index=False)
        except Exception as e:
            logger.error(f"Failed to save small dataset: {e}")

if __name__ == "__main__":
    raw_data_path = os.path.join("data", "raw", "complaints.csv")
    preprocess_data(raw_data_path)

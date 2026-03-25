"""
Compute data quality scores using the trained RandomForest classifier.
Produces a 0-100 quality score for any dataset.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from features.quality_features import extract_quality_features


def load_quality_model():
    """Load the trained RandomForest quality classifier model."""
    model_path = Path("models/quality_model.pkl")
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            "Run: jupyter notebook models/train_quality_model.ipynb"
        )
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    return model


def compute_quality_score(df: pd.DataFrame) -> int:
    """
    Compute a 0-100 quality score for a dataset.
    
    Uses RandomForestClassifier to predict probability of good quality:
    - Score 0-30: POOR quality
    - Score 30-60: FAIR quality  
    - Score 60-85: GOOD quality
    - Score 85-100: EXCELLENT quality
    
    Args:
        df: DataFrame to score
        
    Returns:
        Quality score as integer from 0-100
    """
    
    # Extract 6 quality features
    features = extract_quality_features(df)  # Shape (1, 6)
    
    # Load model
    model = load_quality_model()
    
    # Get probability predictions
    # predict_proba returns [P(bad), P(good)]
    # We want P(good quality) = predict_proba[:, 1]
    quality_probability = model.predict_proba(features)[0, 1]
    
    # Convert probability to 0-100 scale
    quality_score = int(quality_probability * 100)
    
    return quality_score


def compute_quality_report(df: pd.DataFrame) -> dict:
    """
    Compute detailed quality report for a dataset.
    
    Returns:
        Dictionary with quality score, prediction probabilities, and feature breakdown
    """
    
    features = extract_quality_features(df)
    feature_names = [
        "missing_ratio",
        "duplicate_ratio",
        "numeric_ratio",
        "constant_columns",
        "avg_variance",
        "avg_skewness"
    ]
    
    model = load_quality_model()
    
    # Get probabilities
    probabilities = model.predict_proba(features)[0]
    quality_score = compute_quality_score(df)
    
    # Determine rating
    if quality_score >= 85:
        rating = "EXCELLENT"
    elif quality_score >= 60:
        rating = "GOOD"
    elif quality_score >= 30:
        rating = "FAIR"
    else:
        rating = "POOR"
    
    report = {
        "quality_score": quality_score,
        "rating": rating,
        "probability_bad": float(probabilities[0]),
        "probability_good": float(probabilities[1]),
        "features": {
            name: float(val) 
            for name, val in zip(feature_names, features[0])
        },
        "feature_importance": {
            name: float(importance)
            for name, importance in zip(feature_names, model.feature_importances_)
        }
    }
    
    return report


if __name__ == "__main__":
    # Example usage
    print("Quality Score Computation Module")
    print("=" * 50)
    print("\nUsage:")
    print("  from utils.compute_quality_score import compute_quality_score, compute_quality_report")
    print("  import pandas as pd")
    print("  df = pd.read_csv('data.csv')")
    print()
    print("  # Simple score (0-100)")
    print("  score = compute_quality_score(df)")
    print("  print(f'Quality Score: {score}/100')")
    print()
    print("  # Detailed report with probabilities and feature importance")
    print("  report = compute_quality_report(df)")
    print("  print(f'Rating: {report[\"rating\"]}')")
    print("  print(f'P(good): {report[\"probability_good\"]:.2%}')")
    print()
    print("Ratings:")
    print("  85-100: EXCELLENT quality")
    print("  60-85:  GOOD quality")
    print("  30-60:  FAIR quality")
    print("  0-30:   POOR quality")

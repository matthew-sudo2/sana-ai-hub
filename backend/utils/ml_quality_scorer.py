"""
ML-based Data Quality Scoring
Loads trained Random Forest model and extracts features from datasets.
Provides quality assessment scores and predictions.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


class MLQualityScorer:
    """
    Load the trained best_model.pkl and score datasets.
    Features: [missing_ratio, duplicate_ratio, numeric_ratio, constant_cols, 
               norm_variance, skewness, cardinality_ratio, mean_kurtosis]
    """
    
    def __init__(self, model_path: str = "models/best_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained Random Forest model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
    
    @staticmethod
    def extract_features(df: pd.DataFrame) -> list[float]:
        """
        Extract 8 features from a dataframe.
        Returns: [missing_ratio, duplicate_ratio, numeric_ratio, constant_cols, 
                  norm_variance, skewness, cardinality_ratio, mean_kurtosis]
        """
        # Feature 1: Missing data ratio
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        # Feature 2: Duplicate row ratio
        duplicate_ratio = 1 - (len(df.drop_duplicates()) / len(df)) if len(df) > 0 else 0
        
        # Feature 3: Numeric column ratio
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_ratio = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
        
        # Feature 4: Constant columns (only 1 unique value)
        constant_cols = sum(1 for col in df.columns if df[col].nunique() <= 1)
        
        # Feature 5: Normalized variance (Coefficient of Variation)
        cv_list = []
        for col in numeric_cols:
            try:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if abs(mean_val) > 1e-10:
                    cv = min(abs(std_val / mean_val), 100)
                    cv_list.append(cv)
            except:
                pass
        norm_variance = np.mean(cv_list) if cv_list else 0
        
        # Feature 6: Skewness
        skewness_list = []
        for col in numeric_cols:
            try:
                skew = df[col].skew()
                if not pd.isna(skew):
                    skewness_list.append(abs(skew))
            except:
                pass
        skewness = np.mean(skewness_list) if skewness_list else 0
        
        # Feature 7: Cardinality ratio (avg unique values / rows)
        cardinalities = [df[col].nunique() for col in df.columns]
        avg_cardinality = np.mean(cardinalities) if cardinalities else 0
        cardinality_ratio = (avg_cardinality / len(df)) if len(df) > 0 else 0
        cardinality_ratio = min(cardinality_ratio, 1.0)
        
        # Feature 8: Mean Kurtosis (distribution shape)
        kurtosis_list = []
        for col in numeric_cols:
            try:
                kurt = df[col].kurtosis()
                if not pd.isna(kurt):
                    kurtosis_list.append(abs(kurt))
            except:
                pass
        mean_kurtosis = np.mean(kurtosis_list) if kurtosis_list else 0
        mean_kurtosis = min(mean_kurtosis / 10, 1.0)
        
        return [
            missing_ratio, duplicate_ratio, numeric_ratio, constant_cols, 
            norm_variance, skewness, cardinality_ratio, mean_kurtosis
        ]
    
    def score(self, df: pd.DataFrame) -> dict[str, float | int | list]:
        """
        Score a dataset using the ML model.
        
        Returns:
        {
            "quality": "GOOD" or "BAD",
            "score": 0-100 (confidence),
            "probability_good": 0.0-1.0,
            "probability_bad": 0.0-1.0,
            "features": [8 float values]
        }
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if len(df) == 0:
            return {
                "quality": "BAD",
                "score": 0.0,
                "probability_good": 0.0,
                "probability_bad": 1.0,
                "features": []
            }
        
        features = self.extract_features(df)
        
        # Predict
        prediction = self.model.predict([features])[0]  # 0=BAD, 1=GOOD
        probabilities = self.model.predict_proba([features])[0]  # [P(BAD), P(GOOD)]
        
        quality = "GOOD" if prediction == 1 else "BAD"
        prob_bad = float(probabilities[0])
        prob_good = float(probabilities[1])
        score = prob_good * 100 if prediction == 1 else prob_bad * 100
        
        return {
            "quality": quality,
            "score": round(score, 1),
            "probability_good": round(prob_good, 4),
            "probability_bad": round(prob_bad, 4),
            "features": features
        }

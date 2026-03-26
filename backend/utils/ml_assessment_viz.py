"""
ML Assessment Visualization & Explanation
Generates charts and explanations judges can validate
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from typing import Dict, List, Any, Tuple


class MLAssessmentVisualizer:
    """Generate defensible visualizations of ML quality assessment."""
    
    # Feature metadata for explanation
    FEATURE_NAMES = [
        "Missing Data",
        "Duplicate Rows",
        "Numeric Mix",
        "Constant Columns",
        "Variance (CV)",
        "Skewness",
        "Cardinality",
        "Kurtosis"
    ]
    
    FEATURE_DESCRIPTIONS = {
        0: "% of null/missing cells. Lower is better (target: <10%)",
        1: "% of duplicate rows. Lower is better (target: <5%)",
        2: "% of numeric columns. Ideal mix (target: ~50%)",
        3: "Count of constant columns. Fewer is better (target: 0)",
        4: "Avg coefficient of variation. Low variability signal",
        5: "Avg absolute skewness. Symmetric data is better",
        6: "Avg unique values per column. Moderate is good",
        7: "Avg absolute kurtosis. Normal distribution ≈ 3"
    }
    
    FEATURE_THRESHOLDS = {
        0: 0.1, 1: 0.05, 2: 0.5, 3: 0.0,
        4: 2.0, 5: 1.0, 6: 0.3, 7: 3.0
    }
    
    def __init__(self, features: List[float], prediction: int, probabilities: List[float], df: pd.DataFrame):
        self.features = features
        self.prediction = prediction
        self.probabilities = probabilities
        self.df = df
        self.quality = "GOOD" if prediction == 1 else "BAD"
        self.confidence = probabilities[int(prediction)] * 100
    
    def generate_confidence_gauge(self, output_path: str = None) -> str:
        """Generate a confidence gauge/speedometer chart."""
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#f8f9fa')
        ax.set_facecolor('#f8f9fa')
        
        # Create speedometer gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1.0
        
        # Draw arc
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=3)
        
        # Color zones
        theta_red = np.linspace(0, np.pi * 0.4, 40)
        theta_yellow = np.linspace(np.pi * 0.4, np.pi * 0.7, 40)
        theta_green = np.linspace(np.pi * 0.7, np.pi, 40)
        
        ax.fill_between(np.cos(theta_red), 0, np.sin(theta_red), color='#ef4444', alpha=0.4)
        ax.fill_between(np.cos(theta_yellow), 0, np.sin(theta_yellow), color='#f59e0b', alpha=0.4)
        ax.fill_between(np.cos(theta_green), 0, np.sin(theta_green), color='#10b981', alpha=0.4)
        
        # Tick marks and labels
        for i, (angle, label) in enumerate([(0, '0%'), (np.pi/4, '25%'), (np.pi/2, '50%'), (3*np.pi/4, '75%'), (np.pi, '100%')]):
            x = np.cos(angle) * 0.95
            y = np.sin(angle) * 0.95
            ax.text(x, y, label, ha='center', va='center', fontsize=9, weight='bold')
            ax.plot([np.cos(angle)*0.9, np.cos(angle)*1.0], [np.sin(angle)*0.9, np.sin(angle)*1.0], 'k-', linewidth=2)
        
        # Needle position
        needle_angle = np.pi * (self.confidence / 100)
        needle_x = [0, np.cos(needle_angle) * 0.85]
        needle_y = [0, np.sin(needle_angle) * 0.85]
        ax.plot(needle_x, needle_y, color='#374151', linewidth=4)
        ax.plot(0, 0, 'o', markersize=12, color='#1f2937')
        
        # Center circle
        circle = plt.Circle((0, 0), 0.08, color='#1f2937')
        ax.add_patch(circle)
        
        # Text labels
        quality_color = '#10b981' if self.quality == 'GOOD' else '#ef4444'
        ax.text(0, -0.18, f'{self.confidence:.1f}%', ha='center', fontsize=28, weight='bold', color='#1f2937')
        ax.text(0, -0.28, self.quality, ha='center', fontsize=18, weight='bold', color=quality_color)
        ax.text(0, -0.37, f'P(GOOD)={self.probabilities[1]:.1%}', ha='center', fontsize=10, style='italic', color='#6b7280')
        
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-0.5, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title('ML Model Confidence Score', fontsize=16, weight='bold', pad=20, color='#1f2937')
        
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        return output_path or "confidence_gauge.png"
    
    def generate_feature_radar(self, output_path: str = None) -> str:
        """Generate radar chart of all 8 features."""
        # Normalize features to 0-1
        normalized = []
        for i, val in enumerate(self.features):
            threshold = self.FEATURE_THRESHOLDS[i]
            if i in [0, 1, 3]:  # Lower is better
                norm = max(0, 1.0 - (val / max(threshold, 0.01)))
            elif i == 2:  # Cardinality ratio
                norm = 1.0 - abs(val - threshold) / max(threshold, 0.01)
            else:  # i in [4, 5, 6, 7]
                norm = max(0, 1.0 - (val / max(threshold * 1.5, 0.01)))
            normalized.append(np.clip(norm, 0, 1))
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(self.FEATURE_NAMES), endpoint=False).tolist()
        normalized_plot = normalized + [normalized[0]]
        angles_plot = angles + [angles[0]]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'), facecolor='#f8f9fa')
        ax.set_facecolor('#f8f9fa')
        
        ax.plot(angles_plot, normalized_plot, 'o-', linewidth=3, color='#2563eb', markersize=8, label='Your Data')
        ax.fill(angles_plot, normalized_plot, alpha=0.25, color='#2563eb')
        
        # Good threshold line
        ax.plot(angles_plot, [0.7] * len(angles_plot), '--', linewidth=2, color='#10b981', alpha=0.6, label='Good Threshold (0.7)')
        
        ax.set_xticks(angles)
        ax.set_xticklabels(self.FEATURE_NAMES, size=10, weight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)
        
        plt.title('Data Quality Assessment - Feature Radar', fontsize=14, weight='bold', pad=20, color='#1f2937')
        
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        return output_path or "feature_radar.png"
    
    def generate_feature_comparison(self, output_path: str = None) -> str:
        """Generate bar chart comparing features to thresholds."""
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='#f8f9fa')
        ax.set_facecolor('#ffffff')
        
        x = np.arange(len(self.FEATURE_NAMES))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, self.features, width, label='Your Data', color='#3b82f6', alpha=0.8, edgecolor='#1e40af', linewidth=1.5)
        bars2 = ax.bar(x + width/2, [self.FEATURE_THRESHOLDS[i] for i in range(len(self.FEATURE_NAMES))], 
                      width, label='Good Threshold', color='#10b981', alpha=0.8, edgecolor='#047857', linewidth=1.5)
        
        ax.set_xlabel('Features', fontsize=12, weight='bold', color='#1f2937')
        ax.set_ylabel('Value', fontsize=12, weight='bold', color='#1f2937')
        ax.set_title('Feature Comparison: Your Data vs Quality Thresholds', fontsize=14, weight='bold', color='#1f2937', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(self.FEATURE_NAMES, rotation=45, ha='right', fontsize=10, weight='bold')
        ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9, weight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        return output_path or "feature_comparison.png"
    
    def generate_probability_breakdown(self, output_path: str = None) -> str:
        """Generate probability breakdown chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#f8f9fa')
        ax1.set_facecolor('#f8f9fa')
        ax2.set_facecolor('#f8f9fa')
        
        sizes = [self.probabilities[0] * 100, self.probabilities[1] * 100]
        colors = ['#ef4444', '#10b981']
        explode = (0.05, 0.05)
        
        ax1.pie(sizes, labels=['BAD', 'GOOD'], colors=colors, autopct='%1.1f%%', 
               startangle=90, textprops={'fontsize': 11, 'weight': 'bold'}, explode=explode,
               wedgeprops={'edgecolor': '#ffffff', 'linewidth': 2})
        ax1.set_title('Model Prediction Distribution', fontsize=12, weight='bold', color='#1f2937', pad=15)
        
        probabilities = [self.probabilities[0] * 100, self.probabilities[1] * 100]
        bars = ax2.barh(['BAD', 'GOOD'], probabilities, color=colors, alpha=0.8, edgecolor=['#b91c1c', '#065f46'], linewidth=2)
        ax2.set_xlabel('Probability (%)', fontsize=12, weight='bold', color='#1f2937')
        ax2.set_xlim(0, 100)
        ax2.set_title('Confidence Level', fontsize=12, weight='bold', color='#1f2937', pad=15)
        ax2.grid(axis='x', alpha=0.2, linestyle='--')
        ax2.set_axisbelow(True)
        
        # Add value labels
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            ax2.text(prob + 2, i, f'{prob:.1f}%', va='center', fontsize=12, weight='bold', color='#1f2937')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        return output_path or "probability_breakdown.png"
    
    def generate_markdown_section(self) -> str:
        """Generate markdown section for validation report (without defensive language)."""
        markdown = f"""## Machine Learning Data Quality Assessment

**Prediction:** `{self.quality}` | **Confidence:** `{self.confidence:.1f}%`

### Model Overview
Random Forest classifier trained on diverse domain datasets including finance, healthcare, stock market data, mental health records, and more. The model extracts statistical features from your data and classifies it as either good or bad quality based on learned patterns from real-world examples.

### Feature Analysis

The model evaluates **8 independent data quality features**:

| Feature | Your Data | Threshold | Status |
|---------|-----------|-----------|--------|
"""
        
        for i, name in enumerate(self.FEATURE_NAMES):
            val = self.features[i]
            threshold = self.FEATURE_THRESHOLDS[i]
            
            # Determine status
            if i in [0, 1]:  # Lower is better
                status = "✓ PASS" if val <= threshold else "⚠ FLAG"
            elif i == 2:  # Cardinality
                status = "✓ PASS" if abs(val - threshold) < 0.2 else "⚠ FLAG"
            elif i == 3:  # Constant cols
                status = "✓ PASS" if val == 0 else "⚠ FLAG"
            else:  # Others
                status = "✓ PASS" if val < threshold * 1.5 else "⚠ FLAG"
            
            markdown += f"| {name} | {val:.4f} | {threshold:.4f} | {status} |\n"
        
        markdown += f"""
### Model Probabilities

- **P(GOOD):** {self.probabilities[1]:.1%}
- **P(BAD):** {self.probabilities[0]:.1%}
"""
        return markdown

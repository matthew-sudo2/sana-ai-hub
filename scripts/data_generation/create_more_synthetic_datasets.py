"""
Create multiple synthetic GOOD datasets with varying variance levels.
This balances the training data so the model learns variance is not an indicator of quality.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_datasets():
    """Generate 3 more synthetic datasets with different variance profiles"""
    
    good_dir = Path('data/labeled/good')
    good_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset 1: Bank account transactions (low variance - consistent amounts)
    print("[1] Creating bank_account_transactions.csv (low variance, high completeness)...")
    np.random.seed(1)
    n = 3000
    df1 = pd.DataFrame({
        'account_id': np.arange(1, n + 1),
        'date': pd.date_range('2025-01-01', periods=n, freq='D'),
        'transaction_type': np.random.choice(['Deposit', 'Withdrawal', 'Transfer'], n),
        'amount': np.abs(np.random.normal(500, 100, n)).round(2),  # Tightly distributed
        'balance': np.linspace(10000, 25000, n),  # Very consistent growth
        'status': 'Completed',  # All same value = good
        'branch_code': np.random.choice(['NYC', 'LAX', 'CHI'], n),
    })
    df1.to_csv(good_dir / 'bank_account_transactions.csv', index=False)
    print(f"    ✓ {len(df1)} rows, variance: {df1[['amount', 'balance']].var().mean():.0f}")
    
    # Dataset 2: Student grades (moderate variance, clean structure)
    print("[2] Creating student_grades.csv (moderate variance, consistent structure)...")
    np.random.seed(2)
    n = 4000
    df2 = pd.DataFrame({
        'student_id': np.arange(1000, 1000 + n),
        'class_id': np.random.randint(101, 150, n),
        'math_score': np.random.normal(75, 10, n).clip(0, 100),
        'english_score': np.random.normal(75, 10, n).clip(0, 100),
        'science_score': np.random.normal(75, 10, n).clip(0, 100),
        'gpa': np.random.normal(3.2, 0.5, n).clip(0, 4.0),
        'attendance_pct': np.random.normal(92, 5, n).clip(0, 100),
        'passed': np.random.choice([True, False], n, p=[0.85, 0.15]),
    })
    df2.to_csv(good_dir / 'student_grades.csv', index=False)
    print(f"    ✓ {len(df2)} rows, variance: {df2[['math_score', 'english_score', 'science_score', 'gpa']].var().mean():.0f}")
    
    # Dataset 3: Employee records (high variance features but clean - no missing values)
    print("[3] Creating employee_records.csv (high variance, zero missing)...")
    np.random.seed(3)
    n = 2500
    df3 = pd.DataFrame({
        'emp_id': np.arange(50001, 50001 + n),
        'department': np.random.choice(['IT', 'Finance', 'HR', 'Sales', 'Ops'], n),
        'salary': np.random.lognormal(10.7, 0.6, n).astype(int),  # High variance
        'years_employed': np.random.randint(0, 30, n),
        'performance_rating': np.random.normal(3.5, 0.8, n).clip(1, 5),
        'projects_completed': np.random.randint(0, 50, n),
        'hire_date': pd.date_range('1995-01-01', periods=n, freq='D'),
        'is_active': True,  # All True = good
    })
    df3.to_csv(good_dir / 'employee_records.csv', index=False)
    print(f"    ✓ {len(df3)} rows, variance: {df3[['salary', 'years_employed', 'performance_rating']].var().mean():.0f}")
    
    print("\n" + "=" * 60)
    print("✓ Created 3 synthetic GOOD datasets with varying variance")
    print("=" * 60)
    print("\nDataset Summary:")
    print("  1. bank_account_transactions.csv    - LOW variance (consistent transactions)")
    print("  2. student_grades.csv              - MODERATE variance (grade distributions)")
    print("  3. employee_records.csv            - HIGH variance (salary ranges)")
    print("\nAll have: ZERO missing values, ZERO duplicates, CONSISTENT types")
    print("\nThis teaches the model that quality is about completeness, not variance!")

if __name__ == '__main__':
    create_datasets()

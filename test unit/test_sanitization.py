#!/usr/bin/env python3
"""Test script to verify text sanitization for validation reports."""

from pathlib import Path


def sanitize_detail(detail: str) -> str:
    """Sanitize detail text for safe markdown rendering."""
    if not detail:
        return ""
    # Convert Windows paths to forward slashes for better markdown rendering
    detail = detail.replace("\\", "/")
    # Remove special markdown characters or escape them
    detail = detail.replace("|", "•")  # Replace pipes with bullets
    detail = detail.replace("[", "(").replace("]", ")")  # Replace brackets with parens
    # Clean up Python list/dict representations
    detail = detail.replace("'", "")  # Remove single quotes from list repr
    return detail.strip()


def sanitize_for_markdown(value: str) -> str:
    """Escape special markdown characters for safe table rendering."""
    if value is None:
        return ""
    value = str(value)
    # Escape pipe characters for markdown tables
    value = value.replace("|", "•")
    # Escape leading characters that might cause markdown confusion
    if value.startswith(("#", "-", "*", ">")):
        value = " " + value
    return value


# Test cases
test_cases = [
    # Original problematic inputs
    (r"C:\Users\User\Documents\run_dir\missing_file.json", "C/Users/User/Documents/run_dir/missing_file.json"),
    ("['col1', 'col2', 'col3']", "(col1, col2, col3)"),
    ("analysis=1000, actual=1000", "analysis=1000, actual=1000"),
    ("reported=100•5, actual=98|3", "reported=100•5, actual=98•3"),
    
    # Edge cases  
    ("", ""),
    (None, ""),
    ("  spaces  ", "spaces"),
    
    # Table values
    ("Value|With|Pipes", "Value•With•Pipes"),
    ("#Heading", " #Heading"),
    ("- List item", " - List item"),
    ("*Bold*", " *Bold*"),
]

print("Testing sanitize_detail function:")
print("-" * 70)
for input_val, expected in test_cases:
    result = sanitize_detail(input_val) if input_val is not None else sanitize_detail("")
    status = "✓" if result == expected else "✗"
    print(f"{status} Input: {repr(input_val)}")
    print(f"  Expected: {repr(expected)}")
    print(f"  Got:      {repr(result)}")
    if result != expected:
        print(f"  MISMATCH!")
    print()

print("\nTesting sanitize_for_markdown function:")
print("-" * 70)
markdown_tests = [
    ("normal_value", "normal_value"),
    ("value|with|pipes", "value•with•pipes"),
    ("#heading", " #heading"),
    ("- item", " - item"),
    ("*bold*", " *bold*"),
    ("data|value|123", "data•value•123"),
]

for input_val, expected in markdown_tests:
    result = sanitize_for_markdown(input_val)
    status = "✓" if result == expected else "✗"
    print(f"{status} Input: {repr(input_val)}")
    print(f"  Expected: {repr(expected)}")
    print(f"  Got:      {repr(result)}")
    if result != expected:
        print(f"  MISMATCH!")
    print()

print("\nAll tests completed!")

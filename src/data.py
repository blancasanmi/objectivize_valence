"""Data loading and management."""
import pandas as pd
import os
from typing import Iterator

class CSVIterator:
    """Iterates over sentence pairs from a CSV file one by one."""

    def __init__(self, filepath: str):
        self.df = pd.read_csv(filepath, encoding='utf-8-sig')
        self.index = 0

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> dict:
        if self.index < len(self.df):
            row = self.df.iloc[self.index].to_dict()
            self.index += 1
            return row
        raise StopIteration

    def __len__(self) -> int:
        return len(self.df)


def save_result(output_path: str, row: dict, result: dict) -> None:
    """
    Append a single processed row to the output CSV.
    Creates the file with headers if it doesn't exist yet.
    """
    try:
        # Safely extract with .get() to avoid KeyError
        s1 = result.get("sentence_1", {})
        s2 = result.get("sentence_2", {})
        
        # Handle case where sentence_2 is nested twice by Mistral
        if "sentence_2" in s2:
            s2 = s2["sentence_2"]
            
        pair = result.get("pair", {})

        combined = {
            **row,
            "sentiment_1": s1.get("sentiment", None),
            "valence_1": s1.get("valence", None),
            "sentiment_2": s2.get("sentiment", None),
            "valence_2": s2.get("valence", None),
            "pair_uncertainty": pair.get("uncertainty", None)
        }

    except Exception as e:
        print(f"Error processing result for row {row}: {e}")
        combined = {**row, "sentiment_1": None, "valence_1": None, "sentiment_2": None, "valence_2": None, "uncertainty": None}
    
    df_row = pd.DataFrame([combined])
    write_header = not os.path.exists(output_path)
    df_row.to_csv(output_path, mode='a', header=write_header,
                  index=False, encoding='utf-8-sig')
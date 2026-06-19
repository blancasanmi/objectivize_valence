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
            "valence_1": s1.get("valence", None),
            "arousal_1": s1.get("arousal", None),
            "uncertainty_1": s1.get("uncertainty", None),
            "valence_2": s2.get("valence", None),
            "arousal_2": s2.get("arousal", None),
            "uncertainty_2": s2.get("uncertainty", None),
            "resolution": pair.get("resolution", None), 
            "surprise" : pair.get("surprise", None)
        }

    except Exception as e:
        print(f"Error processing result for row {row}: {e}")
        combined = {**row, "valence_1": None, "arousal_1": None, "uncertainty_1": None, "valence_2": None, "arousal_2": None, "uncertainty_2": None, "resolution": None, "surprise": None}

    df_row = pd.DataFrame([combined])
    write_header = not os.path.exists(output_path)
    df_row.to_csv(output_path, mode='a', header=write_header,
                  index=False, encoding='utf-8-sig')
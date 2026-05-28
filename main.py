"""Main entry point for sentence analysis application."""
from src.data import CSVIterator, save_result
from src.models import MistralModel


if __name__ == "__main__":
    iterator = CSVIterator("data/data.csv")
    model = MistralModel("ministral-3b-latest")

    for row in iterator:    
        result = model.score(row["first_sentence"], row["second_sentence"])
        print(f"Processed row {iterator.index}/{len(iterator)}")
        print(f"Result: {result}")
        save_result("data/output.csv", row, result)
       
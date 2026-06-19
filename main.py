"""Main entry point for sentence analysis application."""
from src.data import CSVIterator, save_result
from src.models import MistralModel


if __name__ == "__main__":
    iterator = CSVIterator("data/combined_valid_sentences.csv")
    model = MistralModel("mistral-medium-3-5")

    # for row in iterator:    
    #     result = model.score(row["first"], row["second"])
    #     print(f"Processed row {iterator.index}/{len(iterator)}")
    #     print(f"Result: {result}")
    #     save_result("data/output.csv", row, result)

    with open("data/mistral_answer.txt", "w", encoding="utf-8") as f:
        f.write(model.query_how("surprise"))
        f.write(model.query_how("uncertainty"))
       
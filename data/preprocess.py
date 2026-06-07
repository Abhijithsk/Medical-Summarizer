from datasets import load_dataset

def load_and_prepare(max_train=10000, max_val=500):
    print("Loading PubMed dataset...")
    dataset = load_dataset("ccdv/pubmed-summarization", "document")

    # Use a subset for faster training
    train = dataset["train"].select(range(max_train))
    val   = dataset["validation"].select(range(max_val))

    print(f"Train samples : {len(train)}")
    print(f"Val samples   : {len(val)}")
    print(f"\nSample article (500 chars):\n{train[0]['article'][:500]}")
    print(f"\nSample abstract:\n{train[0]['abstract'][:300]}")

    return train, val

if __name__ == "__main__":
    load_and_prepare()
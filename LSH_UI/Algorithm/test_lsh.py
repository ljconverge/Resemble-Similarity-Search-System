from lsh import generate_candidates
# Example MinHash signatures for four documents.
# These are sample values for testing, not real document signatures.

signatures = {
    "A": [12, 45, 8, 91, 34, 67, 23, 78],
    "B": [12, 45, 8, 91, 55, 67, 23, 80],
    "C": [19, 22, 63, 17, 44, 51, 90, 11],
    "D": [12, 45, 72, 91, 34, 67, 40, 78]
}
def main():
    candidates, buckets = generate_candidates(
        signatures,
        bands=4
    )

    print("BUCKETS")

    for key, document_ids in sorted(buckets.items()):
        print(key, "->", document_ids)

    # Display the candidate document pairs.
    print("\nCANDIDATE PAIRS")

    for pair in sorted(candidates):
        print(pair)

if __name__ == "__main__":
    main()

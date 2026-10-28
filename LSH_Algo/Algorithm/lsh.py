"""Locality-Sensitive Hashing (LSH) candidate generation implemented from scratch."""

from itertools import combinations


def generate_candidates(signatures, bands=20):
    """
    Build LSH buckets and return unique candidate document pairs.

    Args:
        signatures: Mapping of document ID -> MinHash signature (sequence of integers).
        bands: Number of equal-sized bands in each signature.

    Returns:
        (candidates, buckets):
          candidates: set of sorted 2-tuples of document IDs.
          buckets: mapping of (band_index, band_tuple) -> list of document IDs.

    Every signature must have the same non-zero length, and the signature length
    must be divisible by `bands`.
    """
    if not signatures:
        return set(), {}

    if not isinstance(bands, int) or isinstance(bands, bool):
        raise ValueError("The number of bands must be an integer.")

    first_signature = next(iter(signatures.values()))
    signature_length = len(first_signature)
    if signature_length == 0:
        raise ValueError("Signatures cannot be empty.")
    if not 1 <= bands <= signature_length:
        raise ValueError(f"bands must be between 1 and {signature_length}.")
    if signature_length % bands != 0:
        raise ValueError(
            f"Signature length ({signature_length}) must be divisible by bands ({bands})."
        )

    rows_per_band = signature_length // bands
    buckets = {}

    for document_id, signature in signatures.items():
        if len(signature) != signature_length:
            raise ValueError("All signatures must have equal length.")

        for band_index in range(bands):
            start = band_index * rows_per_band
            end = start + rows_per_band
            band_values = tuple(int(value) for value in signature[start:end])
            bucket_key = (band_index, band_values)
            buckets.setdefault(bucket_key, []).append(document_id)

    candidates = set()
    for document_ids in buckets.values():
        if len(document_ids) > 1:
            for pair in combinations(sorted(document_ids), 2):
                candidates.add(pair)

    return candidates, buckets

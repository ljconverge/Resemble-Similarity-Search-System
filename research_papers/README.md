## 1. Frame the problem: exact near-duplicate / similarity search

Start by defining what you're solving: given N items (documents, questions, images), find all pairs whose similarity (Jaccard, cosine, Hamming, edit distance) exceeds some threshold — **without O(N²) pairwise comparisons**.

**Foundational framing papers:**
- Broder, A. (1997). *"On the Resemblance and Containment of Documents"* — SEQUENCES '97. Defines resemblance/containment and the shingling approach that everything else builds on.
- Broder, A., Glassman, S., Manasse, M., Zweig, G. (1997). *"Syntactic Clustering of the Web"* — Computer Networks. The original AltaVista near-duplicate detection system; a great "here's the real-world problem" citation.
- Manku, G.S., Jain, A., Sarma, A.D. (2007). *"Detecting Near-Duplicates for Web Crawling"* — WWW '07. Google's SimHash-based system at web scale; excellent for motivating scale.

## 2. Establish the brute-force baseline and why it breaks down

Explain pairwise comparison cost (O(N²) comparisons, each itself O(d) for feature vectors), then bring in the **curse of dimensionality** — the real reason exact nearest-neighbor methods (KD-trees, etc.) also degrade in high dimensions, motivating approximate methods.

- Indyk, P., Motwani, R. (1998). *"Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality"* — STOC '98. **This is the seminal LSH paper.** Use it both to explain why exact NN search fails in high-D and to introduce LSH as the fix.
- Beyer, K., Goldstein, J., Ramakrishnan, R., Shaft, U. (1999). *"When Is 'Nearest Neighbor' Meaningful?"* — ICDT '99. A classic on why distance concentration breaks exact NN in high dimensions — very citable for your "why brute force / exact indexing structures fail" section.

## 3. Alternate/intermediate approaches (before you land on LSH)

Briefly survey what people tried before or alongside LSH, so LSH doesn't appear out of nowhere:
- **Tree-based exact structures** (KD-trees, R-trees) — mention only to say they degrade to linear scan in high dimensions.
- **Inverted-index / prefix-filtering methods** for set similarity joins — Koudas, N., Srivastava, D. (2005). *"Approximate Joins: Concepts and Techniques"* — VLDB tutorial.
- **Sorted neighborhood / blocking methods** from the record-linkage/deduplication literature (classic pre-LSH dedup heuristic) — useful contrast if your domain is questions/tabular text.

## 4. LSH itself — the core theory

- Indyk & Motwani (1998), as above — the (r, cr, p1, p2)-sensitive hash family definition and the AND–OR (bands and rows) construction.
- Gionis, A., Indyk, P., Motwani, R. (1999). *"Similarity Search in High Dimensions via Hashing"* — VLDB '99. The practical, implementation-oriented LSH paper — probably your most directly useful "how to actually build it" reference.
- Datar, M., Immorlica, N., Indyk, P., Mirrokni, V. (2004). *"Locality-Sensitive Hashing Scheme Based on p-Stable Distributions"* — SoCG '04. LSH for Euclidean (ℓ2) space — cite this if you do image feature vectors under L2/cosine.

## 5. The specific hash families you'll likely implement

- **MinHash (for Jaccard similarity / text shingles):**
  Broder (1997) above, plus Broder, Charikar, Frieze, Mitzenmacher (1998). *"Min-Wise Independent Permutations"* — STOC '98 (the theoretical justification for MinHash).
- **SimHash (for cosine similarity):**
  Charikar, M. (2002). *"Similarity Estimation Techniques from Rounding Algorithms"* — STOC '02. This is the SimHash paper — directly relevant if you're doing text embeddings or want a second family to compare against MinHash.
- Manku et al. (2007), above — SimHash applied at Google web-crawl scale, with a nice discussion of bucket/banding tradeoffs.

## 6. Applying LSH to images

- Perceptual hashing survey — search for "perceptual image hashing survey near-duplicate detection" once you're at that stage; there are several good 2010s surveys (e.g., Zauner, C. (2010), *"Implementation and Benchmarking of Perceptual Image Hashing Schemes"*, master's thesis — very implementation-oriented, good if you want pHash/dHash details) worth pulling up directly.
- If you use CNN embeddings + LSH instead of raw perceptual hashes, cite Datar et al. (2004) again for the p-stable projection scheme, since embeddings are dense real vectors under cosine/L2.

## 7. Survey papers (good for your "related work" table)

- Jafari, O., Maurya, P., Nagarkar, P., Islam, K.M., Crushev, C. (2021). *"A Survey on Locality Sensitive Hashing Algorithms and Their Applications"* — arXiv:2102.08942. Broad, recent, covers many LSH variants — a great single paper to structure your whole related-work section around.
- Wang, J., Shen, H.T., Song, J., Ji, J. (2014). *"Hashing for Similarity Search: A Survey"* — arXiv. Covers both LSH and learning-to-hash methods, useful if you want to gesture at learned alternatives (e.g., product quantization, deep hashing) as future work.

## 8. Speed-vs-accuracy evaluation framing

For your evaluation methodology (comparing brute force vs. LSH on speed/recall/precision), it's worth citing:
- Lv, Q., Josephson, W., Wang, Z., Charikar, M., Li, K. (2007). *"Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search"* — VLDB '07. Great for discussing the recall/query-time/space tradeoff and how practitioners tune LSH (number of hash tables L, hash functions per table k).

---

### Suggested reading order
1. Beyer et al. (1999) — motivate the curse of dimensionality
2. Indyk & Motwani (1998) — core LSH theory
3. Gionis, Indyk, Motwani (1999) — practical LSH construction
4. Broder (1997) + Broder et al. (1998) — MinHash for text
5. Charikar (2002) — SimHash
6. Manku et al. (2007) — real-world web-scale application
7. Datar et al. (2004) — LSH for Euclidean/image feature vectors
8. Lv et al. (2007) — tuning and the speed/accuracy tradeoff
9. Jafari et al. (2021) survey — tie it all together

# Near-Duplicate and Similarity Search Using Locality-Sensitive Hashing

## Project Overview

This project implements a **Near-Duplicate and Similarity Search System** using **Locality-Sensitive Hashing (LSH)** to efficiently identify highly similar or near-duplicate documents in a large collection.

Traditional brute-force similarity search compares every possible pair of documents, resulting in a quadratic number of comparisons. The proposed system uses **k-shingling, MinHash, and Locality-Sensitive Hashing** to reduce the search space by identifying a smaller set of potentially similar documents, called candidate pairs. These candidates are then verified using exact **Jaccard similarity**.

The project also compares the LSH-based approach with brute-force similarity search in terms of execution time, number of comparisons, candidate reduction, speedup, and similarity detection accuracy.

## Objectives

* Implement text preprocessing for uploaded documents.
* Generate k-shingles from documents.
* Calculate exact Jaccard similarity.
* Implement MinHash from scratch.
* Implement Locality-Sensitive Hashing from scratch.
* Generate candidate document pairs using LSH buckets.
* Verify candidate pairs using exact Jaccard similarity.
* Support a user-defined similarity threshold.
* Implement brute-force similarity search as a baseline.
* Compare LSH and brute-force approaches.
* Provide a simple web interface for document upload and analysis.
* Visualize similarity and performance results.

## Core Concept

The system follows the following pipeline:

```text
Documents
    |
    v
Text Preprocessing
    |
    v
K-Shingling
    |
    v
MinHash
    |
    v
LSH Banding
    |
    v
Hash Buckets
    |
    v
Candidate Pairs
    |
    v
Exact Jaccard Similarity
    |
    v
Similarity Threshold
    |
    v
Near-Duplicate Results
    |
    v
Performance Comparison
```

The fundamental idea is to avoid comparing every document with every other document. LSH is first used to identify promising candidate pairs, after which exact similarity verification is performed only on those candidates.

## System Architecture

The project consists of four major layers.

### 1. Presentation Layer

The presentation layer provides a simple interface for interacting with the system.

**Technologies:**

* HTML
* CSS
* JavaScript
* Flask Templates

**Responsibilities:**

* Upload multiple documents
* Configure similarity parameters
* Start similarity analysis
* Display detected near-duplicates
* Display similarity scores
* Display performance statistics

### 2. Application / Backend Layer

The backend connects the web interface with the algorithmic components.

**Technology:**

* Flask

**Responsibilities:**

* Receive uploaded documents
* Manage document collections
* Process user settings
* Invoke algorithm modules
* Manage application routes
* Return results to the frontend

**Main application:**

```text
app.py
```

### 3. Algorithm Layer

The algorithm layer contains the core DSA implementation.

```text
Text Processing
      |
      v
Shingling
      |
      v
Jaccard Similarity
      |
      v
MinHash
      |
      v
Locality-Sensitive Hashing
      |
      v
Candidate Generation
      |
      v
Similarity Verification
```

The core MinHash and LSH algorithms are implemented from scratch rather than using an external LSH library.

### 4. Data Layer

The data layer manages:

* Documents
* Document IDs
* Tokens
* Shingles
* MinHash signatures
* LSH buckets
* Candidate pairs
* Similarity results
* Performance statistics

Python data structures such as **lists, sets, dictionaries, and tuples** are used according to the requirements of each operation.

## Algorithms

### 1. Text Preprocessing

Before similarity analysis, documents are normalized to reduce irrelevant differences.

The preprocessing pipeline is:

```text
Raw Text
   |
   v
Convert to Lowercase
   |
   v
Remove Unnecessary Punctuation
   |
   v
Normalize Whitespace
   |
   v
Tokenization
   |
   v
Processed Text
```

For example:

```text
"Machine Learning is Useful!"
```

may become:

```text
["machine", "learning", "is", "useful"]
```

### 2. K-Shingling

Shingling divides a document into overlapping sequences of words.

For example, using 2-word shingles:

```text
Machine learning is useful
```

produces:

```text
machine learning
learning is
is useful
```

The shingles are stored as a set to eliminate duplicate shingles.

### 3. Jaccard Similarity

The exact similarity between two documents is calculated using Jaccard similarity:

$$
J(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

where:

* `A` represents the shingles of the first document.
* `B` represents the shingles of the second document.
* `A ∩ B` represents common shingles.
* `A ∪ B` represents all unique shingles.

The similarity score ranges from:

```text
0 → Completely different
1 → Identical
```

Jaccard similarity is also used for the final verification of LSH candidate pairs.

### 4. MinHash

Directly comparing large shingle sets can become computationally expensive. MinHash creates a compact signature for each document while approximately preserving Jaccard similarity.

Conceptually:

```text
Large Shingle Set
       |
       v
    MinHash
       |
       v
Compact Signature
```

For example:

```text
Document 1
    |
    v
[12, 45, 8, 91, 34, 67, 23, 78]
```

The key MinHash property is:

$$
P(h(A)=h(B)) \approx J(A,B)
$$

Thus, documents with high Jaccard similarity are likely to have similar MinHash signatures.

The MinHash implementation is developed from scratch.

### 5. Locality-Sensitive Hashing

Locality-Sensitive Hashing is used to efficiently identify potentially similar documents.

The MinHash signature is divided into multiple bands.

For example:

```text
MinHash Signature
[12, 45, 8, 91, 34, 67, 23, 78]

Band 1 → [12, 45]
Band 2 → [8, 91]
Band 3 → [34, 67]
Band 4 → [23, 78]
```

Each band is hashed into a bucket.

Documents whose corresponding bands produce the same bucket are treated as candidate pairs.

```text
Band
 |
 v
Hash Function
 |
 v
Hash Bucket
 |
 v
Candidate Documents
```

Similar documents have a higher probability of being placed in the same bucket.

## Candidate Generation and Verification

Being placed in the same LSH bucket does **not** mean that two documents are definitely duplicates.

It means that they are promising candidates for further verification.

The process is:

```text
Same LSH Bucket
       |
       v
Candidate Pair
       |
       v
Exact Jaccard Similarity
       |
       v
Threshold Check
       |
       +------ Similarity >= Threshold ------> Near Duplicate
       |
       +------ Similarity < Threshold --------> Not Near Duplicate
```

For example, if the similarity threshold is `0.80`:

```text
Similarity = 0.91
0.91 >= 0.80
Result: Near Duplicate
```

Whereas:

```text
Similarity = 0.72
0.72 < 0.80
Result: Not a Near Duplicate
```

## Brute-Force Baseline

To evaluate the effectiveness of LSH, the project implements a brute-force approach.

For `n` documents, the number of possible document pairs is:

$$
\frac{n(n-1)}{2}
$$

For example:

| Documents | Possible Pairs |
| --------: | -------------: |
|     1,000 |        499,500 |
|    10,000 |     49,995,000 |

The brute-force method calculates the exact similarity for every possible pair.

```text
All Documents
      |
      v
Generate Every Possible Pair
      |
      v
Exact Jaccard Similarity
      |
      v
Threshold Check
      |
      v
Results
```

## LSH-Based Approach

The LSH approach reduces the number of comparisons:

```text
All Documents
      |
      v
Shingling
      |
      v
MinHash
      |
      v
LSH Banding
      |
      v
Hash Buckets
      |
      v
Candidate Pairs
      |
      v
Exact Verification
      |
      v
Results
```

Instead of checking every possible pair, only promising candidates generated by LSH are verified.

## Performance Evaluation

The system evaluates both approaches using the following metrics.

### Execution Time

Measures how long each approach takes to process the document collection.

### Number of Comparisons

Measures the number of document pairs actually checked.

### Candidate Reduction

Measures how much the number of candidate pairs is reduced compared with brute force.

### Speedup

$$
Speedup =
\frac{Brute\ Force\ Time}{LSH\ Time}
$$

### Accuracy / Recall

Measures how effectively the LSH approach identifies genuinely similar documents compared with the brute-force baseline.

The project will also evaluate different LSH configurations to demonstrate the trade-off between search speed and similarity-detection accuracy.

## Data Structures

Since this is a DSA-focused project, the implementation explicitly uses and justifies appropriate data structures.

| Data Structure    | Purpose                                          |
| ----------------- | ------------------------------------------------ |
| List              | Store ordered tokens and MinHash signatures      |
| Set               | Store unique shingles and perform set operations |
| Dictionary        | Map document IDs and LSH buckets to data         |
| Tuple             | Represent immutable values and bucket keys       |
| Hash Table        | Provide efficient LSH bucket lookup              |
| 2D Array / Matrix | Store MinHash signatures                         |
| Set of Pairs      | Store unique candidate document pairs            |

The selection of each data structure is based on the operations required by the corresponding algorithm.

## Web Application

The project provides a simple Flask-based web interface.

### Planned Features

#### Document Upload

Upload multiple documents and process them as a single collection.

#### Similarity Threshold

Allow the user to specify the required similarity level.

Example:

```text
Similarity Threshold: 80%
```

#### Similarity Analysis

Start the LSH-based similarity search and generate candidate pairs.

#### Results

Display:

```text
Document A
Document B
Similarity Score
Classification
```

Example:

```text
Document A: question_01.txt
Document B: question_15.txt
Similarity: 91%
Status: Near Duplicate
```

#### Performance Dashboard

Display the comparison between brute-force and LSH approaches, including:

* Execution time
* Number of comparisons
* Candidate pairs
* Candidate reduction
* Speedup
* Accuracy / Recall

## Project Structure

```text
LSH-Similarity-Search/
|
├── app.py
├── requirements.txt
├── README.md
|
├── algorithms/
│   ├── preprocessing.py
│   ├── shingling.py
│   ├── similarity.py
│   ├── minhash.py
│   ├── lsh.py
│   ├── brute_force.py
│   └── verification.py
|
├── data/
│   └── uploads/
|
├── templates/
│   ├── index.html
│   ├── results.html
│   └── dashboard.html
|
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
|
├── evaluation/
│   └── benchmark.py
|
└── tests/
    ├── test_shingling.py
    ├── test_minhash.py
    └── test_lsh.py
```

## Technology Stack

### Programming Language

**Python**

### Backend Framework

**Flask**

### Frontend

* HTML
* CSS
* JavaScript

### Core Algorithms

* K-Shingling
* Jaccard Similarity
* MinHash
* Locality-Sensitive Hashing
* Brute-Force Similarity Search

### Core Data Structures

* Lists
* Sets
* Dictionaries
* Tuples
* Hash Tables
* Matrices

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd LSH-Similarity-Search
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

The application will then be available locally.

## Example

Consider the following documents:

```text
D1: Machine learning is useful for healthcare.

D2: Machine learning is very useful for healthcare.

D3: The capital of Gujarat is Gandhinagar.
```

After preprocessing and shingling, the system generates a set of shingles for each document.

MinHash then produces compact signatures:

```text
D1 → [ ... ]
D2 → [ ... ]
D3 → [ ... ]
```

LSH divides these signatures into bands and hashes them into buckets.

If D1 and D2 share an appropriate bucket, they become a candidate pair:

```text
D1 ↔ D2
```

The candidate pair is then verified using exact Jaccard similarity.

If:

```text
Similarity >= Threshold
```

the pair is classified as a near duplicate.

## Additional Features

The following features may be included as extensions:

### Similarity Search Dashboard

Allow users to inspect the most similar documents in the collection.

### Brute Force vs LSH Comparison

Provide a direct comparison between exact brute-force search and approximate LSH search.

### Explainable Results

For each detected pair, display:

* Similarity score
* Common shingles
* Shared LSH buckets
* Candidate status
* Final classification

These features make the algorithmic process easier to understand during project demonstration and viva.

## Evaluation Criteria

The project will be evaluated based on:

### Correctness

Whether similarity calculation, candidate generation, and duplicate detection operate correctly.

### Data Structure Selection

Whether appropriate data structures are selected for different operations and whether their selection is properly justified.

### Implementation

Whether MinHash and LSH are correctly implemented from scratch.

### Algorithmic Understanding

The team should be able to explain:

* Shingling
* Jaccard similarity
* MinHash
* LSH
* LSH banding
* Hash buckets
* Candidate generation
* Similarity thresholding
* Brute-force comparison

### Overall Functionality

Whether the complete system runs successfully on a local machine and provides meaningful similarity and performance results.

## Important Implementation Note

The core **MinHash and Locality-Sensitive Hashing algorithms will be implemented from scratch** rather than relying on an external LSH library.

External Python libraries may be used for supporting functionality such as:

* Web development
* File handling
* Text preprocessing
* Visualization
* Testing

However, the central DSA implementation will remain transparent, understandable, and explainable by the project team.

## Learning Goals

The project is designed to provide practical understanding of how data structures and algorithms can be applied to large-scale similarity search.

The team should be able to explain the complete process:

```text
Why is brute-force comparison expensive?
              |
              v
What are shingles?
              |
              v
How is Jaccard similarity calculated?
              |
              v
Why is MinHash required?
              |
              v
How does MinHash approximate Jaccard similarity?
              |
              v
How does LSH reduce the search space?
              |
              v
What are bands and buckets?
              |
              v
What is a candidate pair?
              |
              v
Why is exact verification required?
              |
              v
How is the performance improvement measured?
```

The ultimate goal is to develop a functional similarity-search system while demonstrating a clear understanding of the underlying algorithms, data structures, and speed-versus-accuracy trade-offs.

## Team

**Group 2**

| Name              | Student ID |
| ----------------- | ---------- |
| Diya Tilwani      | 202618011  |
| Vedant Kapur      | 202618061  |
| Bhumi Halatwala   | 202618038  |
| Devanshi Dudhatra | 202618027  |
| Dhruv Bhanushali  | 202618053  |
| Lakshita Pagaria  | 202618052  |
| Vrunda Joshi      | 202518043  |

---
date: 2025-01-20
comments: true
tags:
    - NLP
    - Information Retrieval
    - RAG
categories:
    - Information Retrieval
---

# BM25 Explained: A Better Ranking Algorithm than TF-IDF

BM25 algorithm is a popular ranking function used in information retrieval tasks such as search engines. However, BM25 search has also become increasingly popular for RAG (Retrieval Augmented Generation) based systems for ranking documents based on their relevance to a query.

BM25 search is an improved version of the TF-IDF algorithm that addresses some of its limitations. In this article, we will explore the BM25 algorithm in detail, understand its components, compare it with TF-IDF, and implement it from scratch. But before diving into BM25, it is **essential** to understand the nitigrities of TF-IDF, which you can find in my [previous article](./tf_idf_explained.md).

<!-- more -->

## Understanding BM25

As previously mentioned, BM25 is an improved version of the TF-IDF algorithm that addresses some of its limitations. BM25 stands for **Best Matching 25**. So naturally the formula for BM25 is a bit similar to TF-IDF but with some modifications. This formula also contains the term frequency (TF) and inverse document frequency (IDF) components, but it introduces additional parameters to control the term frequency saturation and document length normalization.

### BM25 Formula

$$
\text{BM25}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f_{t, d} \cdot (k_1 + 1)}{f_{t, d} + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}
$$

#### Where

1. **$t$**: A term in the query.
2. **$f_{t, d}$**: The frequency of term $t$ in document $d$ (term frequency).
3. **$|d|$**: The length of document $d$ (number of terms in $d$).
4. **$\text{avgdl}$**: The average document length in the corpus.
5. **$\text{IDF}(t)$**: The inverse document frequency of term $t$, calculated as:
   $$
   \text{IDF}(t) = \log\left(\frac{N - n_t + 0.5}{n_t + 0.5} + 1\right)
   $$
   - $N$: Total number of documents.
   - $n_t$: Number of documents containing term $t$.
6. **$k_1$**: Controls the term frequency saturation (commonly set to 1.2).
7. **$b$**: Controls the document length normalization (commonly set to 0.75).

### Why BM25 is better than tf-idf?

#### Limitations of TF-IDF

While TF-IDF is a good measure for information retrieval, it has some limitations.

**1. Lack of Saturation for Term Frequency:** TF-IDF does not have a saturation point for term frequency. This means that the score for a document keeps increasing linearly with the increase in the frequency of a term in the document. For example, a document that has the term `apple` 15 times may not be much more relevant than a document that has the term `apple` 5 times. However, the difference in the score between these two documents will be significant in TF-IDF.

**2. Bias for longer documents:** TF-IDF has a bias towards longer documents. This is because longer documents tend to have more terms and hence a higher term frequency. This can lead to longer documents being ranked higher than shorter documents even if the shorter documents are more relevant.

**3. Unbounded values for IDF:** Rarer terms have a higher IDF value, which can lead to unbounded scores for these terms. This can skew the ranking of documents towards rare terms, which may not be the most relevant document to the query.

#### How BM25 Addresses These Limitations

**1. Adding Saturation for Term Frequency:** BM25 introduces a saturation point for term frequency by using the term frequency saturation parameter $k_1$. This parameter controls how quickly the score increases with the increase in term frequency. By setting an appropriate value for $k_1$, we can ensure that the score does not increase linearly with term frequency and reaches a saturation point.

**Example:**

Imagine two documents about fruits:
Document 1: Mentions "apple" 3 times.
Document 2: Mentions "apple" 15 times.

**With $k_1 = 1.5$:**

- For Document 1:
  $$
  \text{TF factor} = \frac{3}{3 + 1.5} = \frac{3}{4.5} \approx 0.67
  $$
- For Document 2:
  $$
  \text{TF factor} = \frac{15}{15 + 1.5} = \frac{15}{16.5} \approx 0.91
  $$

Notice that the jump from 3 to 15 repetitions increases the score, but **not proportionally**.

**With $k_1 = 0.5$:**

- For Document 1:
  $$
  \text{TF factor} = \frac{3}{3 + 0.5} = \frac{3}{3.5} \approx 0.86
  $$
- For Document 2:
  $$
  \text{TF factor} = \frac{15}{15 + 0.5} = \frac{15}{15.5} \approx 0.97
  $$

With a smaller $k_1$, the effect of term frequency is further reduced, and the **scores for the two documents become closer**.

**2. Length Normalization:** BM25 introduces a length normalization parameter $b$ that helps to address the bias towards longer documents.  It adjusts the importance of term frequency normalization based on the length of a document compared to the average document length. Here's how it works:

- **$b = 0$**: No length normalization is applied. Long and short documents are treated equally in terms of their length.

- **$b = 1$**: Full length normalization is applied. Term frequency is scaled entirely by the document's length relative to the average length.

- **Intermediate values (e.g., $b = 0.75$)**: Partial length normalization is applied (this is the typical default). It balances the influence of document length on the score, ensuring long documents aren't overly penalized and short ones aren't overly favored.

**3. Adding Smoothing Constants:** BM25 introduces smoothing constants (+0.5) to prevent extreme values for rare terms and balances the weights between rare and frequent terms. As a result, BM25 **avoids overemphasizing rare terms** while still highlighting their importance in a more controlled way. This creates a more balanced and practical scoring mechanism for ranking documents.

**Practical Example:**

**Dataset:**

- $N = 100$ documents
- $n_t = 50$ (term appears in half the documents)

**IDF Comparison:**

- **TF-IDF:**  
  $$
  \text{IDF} = \log\left(\frac{100}{50}\right) = \log(2) \approx 0.693
  $$

- **BM25:**  
  $$
  \text{IDF} = \log\left(\frac{100 - 50 + 0.5}{50 + 0.5} + 1\right) = \log\left(\frac{50.5}{50.5} + 1\right) = \log(2) \approx 0.693
  $$

Here, both are similar because $n_t$ is moderate.

**For Rare Term ($n_t = 1$):**

- **TF-IDF:**  
  $$
  \text{IDF} = \log\left(\frac{100}{1}\right) = \log(100) \approx 4.605
  $$

- **BM25:**  
  $$
  \text{IDF} = \log\left(\frac{100 - 1 + 0.5}{1 + 0.5} + 1\right) = \log\left(\frac{99.5}{1.5} + 1\right) = \log(67.33) \approx 4.208
  $$

BM25 gives a slightly lower score, smoothing the impact of very rare terms.

- **TF-IDF IDF:** Simple but can overemphasize rare terms and lacks robustness.
- **BM25 IDF:** Smoothed and balanced, making it more suitable for real-world search and ranking tasks.

BM25's IDF is tailored for document ranking, where the balance between common and rare terms is crucial for relevance.

## Code implementation of BM25

### Simple implemenation using library

Before we implement BM25 from scratch, let's see how we can use the `rank_bm25` library to calculate BM25 scores for a query and retrieve the top documents based on these scores.

```python
from rank_bm25 import BM25Okapi

corpus = [
    "Apple Apple Banana",
    "Banana Mango Banana",
    "Cherry Cherry Strawberries",
    "Grapes Grapes Strawberries Grapes",
    "Apple Banana Mango",
    "Blueberries Strawberries Apple",
    "Apple Banana Mango",
    "Grapes Grapes Grapes",
    "Blueberries Apple Strawberries",
    "Apple Banana Apple",
    "Cherry Cherry Mango Cherry",
    "Blueberries Strawberries Cherry",
]
tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

query = "banana mango"
tokenized_query = query.lower().split(" ")

doc_scores = bm25.get_scores(tokenized_query)
print(doc_scores)

docs = bm25.get_top_n(tokenized_query, tokenized_corpus, n=5)
docs
```

`BM25Okapi` is a class that implements the BM25 algorithm. It takes a tokenized corpus as input and provides methods to calculate scores for queries and retrieve the top documents based on these scores. The `get_scores` method calculates the BM25 score for each document in the corpus based on the query, while the `get_top_n` method retrieves the top `n` documents based on these scores.

**Output:**

```python
[0.3176789  1.10212021 0.         0.         0.96909597 0.
 0.96909597 0.         0.         0.3176789  0.56864878 0.        ]
[['banana', 'mango', 'banana'],
 ['apple', 'banana', 'mango'],
 ['apple', 'banana', 'mango'],
 ['cherry', 'cherry', 'mango', 'cherry'],
 ['apple', 'banana', 'apple']]
```

### From scratch implementation

#### from scratch module

```python
import math
import numpy as np
from collections import Counter

class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.2, b: float = 0.75):
        tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
        self.k1 = k1
        self.b = b
        self.N = len(tokenized_corpus)
        self.doc_len = []
        self.avg_dl = None
        self.nd, self.document_frequencies = self.initialize(tokenized_corpus)
        self.idf = self.calculate_idf(self.nd)

    def initialize(self, corpus):
        nd = {}
        document_frequencies = []
        total_doc_len = 0
        for doc in corpus:
            doc_len = len(doc)
            self.doc_len.append(doc_len)
            total_doc_len += doc_len
            document_frequencies.append(dict(Counter(doc)))
            for term in set(doc):
                if term not in nd:
                    nd[term] = 1
                else: 
                    nd[term] += 1

        self.avg_dl = total_doc_len / self.N
        return nd, document_frequencies

    def calculate_idf(self, nd):
        idf = {}
        for term in nd:
            idf[term] = math.log((self.N - nd[term] + 0.5) / (nd[term] + 0.5) + 1)

        return idf
    
    def get_scores(self, query: str):
        query = query.lower().split(" ")
        scores = np.zeros(self.N)
        for q in query:
            idf_q = self.idf[q]
            # this is a list of f_q since we are calculating the score for each document 
            f_q = np.array([doc.get(q, 0)  for doc in self.document_frequencies])
            scores += idf_q * (f_q * (self.k1 + 1)) / (f_q + (self.k1 * (1 - self.b + (self.b * np.array(self.doc_len) / self.avg_dl))))

        return scores
    
    def get_top_n(self, query: str, documents: list[str], top_n: int = 5):
        if len(documents) != self.N:
            raise ValueError("The documents do not match the indexed corpus")
        
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:5]
        return [{"doc_id": int(i), "doc": documents[i], "score": round(float(scores[i]), 3)} for i in top_n]
```

The `BM25` class implements the BM25 algorithm from scratch. This class has 3 main methods:

1. `__init__`: Initializes the BM25 model with the corpus and sets the parameters $k_1$ and $b$.
2. `initialize`: This method calculates the `nd` and `document_frequencies`. `nd` is a dictionary that stores the number of documents containing each term, while `document_frequencies` is a list of dictionaries that store the term frequencies for each document. It also calculates the average document length `avg_dl`.
3. `calculate_idf`: This method calculates the IDF values for each term in the corpus based on the formula provided earlier.
4. `get_scores`: This method calculates the BM25 scores for each document in the corpus based on the query. It returns an array of scores for each document. While iterating over the query terms:
   - It gets the IDF value for the term.
   - It calculates the term frequency (f_{t, d}$ in the formula) for the term in each document.
   - It calculates the BM25 score (as per the above formula) for the term in each document and adds it to the total score.
5. `get_top_n`: This method retrieves the top `n` documents based on the BM25 scores. It returns a list of dictionaries containing the document ID, the document text, and the BM25 score for each of the top documents.

#### use module

```python
bm25 = BM25(corpus)
print(f"Term Count: {bm25.nd}")
print(f"Avg Doc Length: {bm25.avg_dl}")
print(f"Document Frequencies: {bm25.document_frequencies}")
```

**Output:**

```python
Term Count: {'apple': 6, 'banana': 5, 'mango': 4, 'strawberries': 5, 'cherry': 3, 'grapes': 2, 'blueberries': 3}
Avg Doc Length: 3.1666666666666665
Document Frequencies: [{'apple': 2, 'banana': 1}, {'banana': 2, 'mango': 1}, {'cherry': 2, 'strawberries': 1}, {'grapes': 3, 'strawberries': 1}, {'apple': 1, 'banana': 1, 'mango': 1}, {'blueberries': 1, 'strawberries': 1, 'apple': 1}, {'apple': 1, 'banana': 1, 'mango': 1}, {'grapes': 3}, {'blueberries': 1, 'apple': 1, 'strawberries': 1}, {'apple': 2, 'banana': 1}, {'cherry': 3, 'mango': 1}, {'blueberries': 1, 'strawberries': 1, 'cherry': 1}]
```

```python
queries = [
    "apple mango",
    "grapes",
    "banana mango",
    "Cherry",
    "apple",
    "Blueberries Strawberries"
]

query = queries[2]

print(f"Query: {query}")
scores = bm25.get_scores(query)
print(f"Scores: {scores}")
docs = bm25.get_top_n(query, corpus)
docs
```

**Output:**

```python
Query: banana mango
Scores: [0.8791299  2.28476434 0.         0.         1.96334623 0.
 1.96334623 0.         0.         0.8791299  0.95776345 0.        ]
[{'doc_id': 1, 'doc': 'Banana Mango Banana', 'score': 2.285},
 {'doc_id': 6, 'doc': 'Apple Banana Mango', 'score': 1.963},
 {'doc_id': 4, 'doc': 'Apple Banana Mango', 'score': 1.963},
 {'doc_id': 10, 'doc': 'Cherry Cherry Mango Cherry', 'score': 0.958},
 {'doc_id': 9, 'doc': 'Apple Banana Apple', 'score': 0.879}]
```

## Conclusion

BM25 is a powerful ranking algorithm that addresses some of the limitations of TF-IDF by introducing term frequency saturation, document length normalization, and smoothing constants. By balancing the importance of term frequency and document length, BM25 provides more robust and relevant document rankings for information retrieval tasks.

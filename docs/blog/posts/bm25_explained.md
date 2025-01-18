# Indepth Understanding of BM25 for Information Retrieval

some intro:

## Understanding BM25

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

**3. Unbounded values for IDF:** Rarer terms have a higher IDF value, which can lead to unbounded scores for these terms. This can skew the ranking of documents towards rare terms, which may not be the most relevant.

#### BM25 as a solution

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

With a smaller $k_1$, the effect of term frequency is further reduced, and the scores for the two documents become closer.

**2. Length Normalization:** BM25 introduces a length normalization parameter $b$ that helps to address the bias towards longer documents.  It adjusts the importance of term frequency normalization based on the length of a document compared to the average document length. Here's how it works:

- **$b = 0$**: No length normalization is applied. Long and short documents are treated equally in terms of their length.

- **$b = 1$**: Full length normalization is applied. Term frequency is scaled entirely by the document's length relative to the average length.

- **Intermediate values (e.g., $b = 0.75$)**: Partial length normalization is applied (this is the typical default). It balances the influence of document length on the score, ensuring long documents aren't overly penalized and short ones aren't overly favored.

**Intuition**:  

- Higher $b$: Shorter documents gain more relevance (useful when query terms are more concentrated in short documents).  
- Lower $b$: Document length has less impact, favoring raw term frequency.

**3. Adding Smoothing Constants:** BM25 introduces smoothing constants (+0.5) to prevent extreme values for rare terms and balances the weights between rare and frequent terms. As a result, BM25 avoids overemphasizing rare terms while still highlighting their importance in a more controlled way. This creates a more balanced and practical scoring mechanism for ranking documents.

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


## Code implementation of BM25

### Simple implemenation using library

### From scratch implementation
#### from scratch module
#### use module
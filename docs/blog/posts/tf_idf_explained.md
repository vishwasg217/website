---
date: 2025-01-01
comments: true
tags:
    - NLP
    - RAG
    - Information Retrieval
categories:
    - Information Retrieval
---

# A Comprehensive Guide on TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) is a popular technique in Natural Language Processing (NLP) for text analysis and information retrieval. It is used to evaluate the importance of a word in a document relative to a collection of documents (corpus). This guide provides an in-depth explanation of TF-IDF, its intuition, mathematical formulation, and implementation from scratch.

<!-- more -->

## Intuition behind TF-IDF

Earlier search methods, and why they were not effective. What TF and IDF bring to the table.

Before TF-IDF, the most common methods for text representation was Bag of Words (BOW). BOW represents text as a collection of words without considering the importance of certain words in context of the document or corpus. This method has some limitations:

1. All terms are treated equally: BOW assigns equal importance to all words in a document. This can be problematic because some words are more important than others.

2. Doesn't account for word rarity: BOW doesn't consider the uniqueness of words across documents. Common words like "the", "is", "and" appear in many documents and are less helpful in distinguishing one document from another.

### How TF IDF overcomes these problems

TF-IDF is a statistical measure that evaluates the importance of a word in a document relative to a collection of documents (corpus). It combines two metrics: Term Frequency (TF) and Inverse Document Frequency (IDF).

Term Frequency (TF): Measures how often a terms appears in a document. Terms that appear more frequently in a document are likely important for that document.

Inverse Document Frequency (IDF): Measures how unique or rare a word is across the entire corpus. If a word appears in many documents, it's less helpful in distinguishing one document from another (e.g., "the"). On the other hand, if a term appears in just a few documents, it's more distinctive.

Combining these two metrics is useful for document search because it helps identify terms that are important to a document. For example, if you're searching for a document about the python programming language, you'd expect the terms "python" and "programming" to appear frequently. By calculating the TF-IDF score, we can identify terms that are important to a document while filtering out common terms like "the", "is", "and", etc. that appear in many documents.

### TF IDF Explained

TF: This measures how often a word appears in a document. Words that appear more frequently in a document are likely important for that document.

IDF: Inverse Document Frequency (IDF): This measures how unique or rare a word is across the entire corpus. If a word appears in many documents, it's less helpful in distinguishing one document from another (e.g., "the"). On the other hand, if a word appears in just a few documents, it's more distinctive.

Combining TF IDF:

$$ TF-IDF = TF * IDF $$

High TF-IDF: The word is frequent in the document but rare in the corpus (important and unique).

Low TF-IDF: The word is either common across the corpus or not frequent in the document (less important).

## Comparing TF IDF with other methods

### Pros

1. **Simple and easy to implement**: TF-IDF is straightforward to implement, making it accessible for various applications in text analysis and NLP.
2. **Effective Keyword Extraction**: It ranks words based on their frequency in a document relative to their frequency across a corpus, which enhances the identification of significant keywords.
3. **Works with large datasets**: The more sophisticated implementations of TF-IDF can handle large datasets efficiently.

### Cons

1. **Biased towards long documents**: TF-IDF can be biased towards longer documents, as they tend to have higher term frequencies.
2. **Does not have semantic understanding**: TF-IDF does not consider the semantic meaning of words, which can lead to misinterpretation in context-rich environments.
3. **No Handling of Synonyms or Polysemy**: While TF-IDF is effective, it may not perform as well in scenarios requiring deeper contextual understanding, such as sentiment analysis or nuanced language processing. It cannot resolve ambiguity in meaning (e.g., "bank" as a financial institution vs. a riverbank).

## The Math Behind TF-IDF

### Mathematical formula

$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$

Where:

#### Term Frequency (TF)

$$
\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}
$$

- $f_{t, d}$: The frequency of term $t$ in document $d$.
- $\sum_{t' \in d} f_{t', d}$: The total number of terms in document $d$.

#### Inverse Document Frequency (IDF)

$$
\text{IDF}(t, D) = \log \left( \frac{|D|}{1 + |d \in D : t \in d|} \right)
$$

- $|D|$: The total number of documents in the collection $D$.
- $|d \in D : t \in d|$: The number of documents in which the term $t$ appears.
- $1$: Added to avoid division by zero in case $t$ does not appear in any document.

---

### Explanation

1. **Where $t$ is the term**:
   - A single word or token for which the TF-IDF score is being calculated.
2. **Where $d$ is the document**:
   - A specific document from the corpus $D$ where the term $t$ is being considered.
3. **Where $D$ is the corpus**:
   - The entire collection of documents.
4. **Where $f_{t, d}$ is the term frequency**:
   - The raw count of term $t$ in document $d$.
5. **Where $\log$ is the logarithm function**:
   - Used to scale down the effect of IDF when a term appears in many documents.

## Building TF-IDF from Scratch

### TF-IDF Vectorizer Implementation

The `TfidfVectorizer` class is implemented in Python to calculate the TF-IDF score for a given corpus of documents. The class has three main methods:

1. `fit(corpus)`: This method calculates the vocabulary and document frequency for the given corpus.
2. `transform(corpus)`: This method calculates the TF-IDF score for each document in the corpus.
3. `fit_transform(corpus)`: This method combines the `fit` and `transform` methods to calculate the TF-IDF score in a single step.

```python
import numpy as np
import pandas as pd
import math

class TfidfVectorizer:

    def __init__(self):
        self.vocabulary = {}
        self.document_frequency = {}
        
    def fit(self, corpus: list[str]): 
        # create vocabulary
        for doc in corpus:
            for word in set(doc.lower().split()):
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)

        # calculate the number of documents in which the terms in the vocabulary appears.
        self.document_frequency = {term: 0 for term in self.vocabulary}
        for doc in corpus:
            unique_terms = set(doc.lower().split())
            for word in unique_terms:
                self.document_frequency[word] += 1

        # calculate the inverse document frequency
        self.inverse_document_frequency = {}
        N = len(corpus)
        for term, df in self.document_frequency.items():
            self.inverse_document_frequency[term] = math.log(N/df+1)
```

The `fit` method does the following:

1. Builds the vocabulary by reading through each document in the `corpus`. It breaks the document into words, converts them to lowercase, and takes only the unique words (using `set()`). It checks if each word is already in self.vocabulary. If not, it adds the word with a unique index (using the current length of the dictionary as the index).
2. `self.document_frequency` is initialized as a dictionary where each word (from the vocabulary) starts with a count of 0. For each document it extracts the unique words. Then, for each unique word in the document, increase its document frequency (DF) by 1.
3. Calculates the Inverse Document Frequency (IDF) for each term in the vocabulary. It uses the formula: $\text{IDF}(t, D) = \log \left( \frac{|D|}{1 + |d \in D : t \in d|} \right)$.

```python
    def transform(self, corpus: list[str]) -> np.ndarray:
        # initialize a matrix of zeros
        tf_idf_matrix = np.zeros((len(corpus), len(self.vocabulary)))

        # count the term frequency for each document
        for i, doc in enumerate(corpus):
            term_count = {}
            for term in doc.lower().split():
                term_count[term] = term_count.get(term, 0) + 1

            # calculate the tf-idf score for each term in the document
            for term, count in term_count.items():
                if term in self.vocabulary:
                    tf = count
                    idf = self.inverse_document_frequency[term]
                    tf_idf_matrix[i][self.vocabulary[term]] = tf * idf

        return tf_idf_matrix 
    
    def fit_transform(self, corpus: list[str]) -> np.ndarray:
        self.fit(corpus)
        result = self.transform(corpus)
        return result
```

The `transform` method does the following:

1. A 2D matrix of zeros is created. Rows = Number of documents in the corpus. Columns = Number of unique words in the vocabulary.
2. Loop through each document in the corpus. For each document, count the frequency of each term, which is basically the Term Frequency (TF).
3. Calculate the TF-IDF score for each term in the document. If the term is in the vocabulary, calculate the TF-IDF score by using the TF values calculated in the previous step and the IDF values calculated during the `fit` method.

The `fit_transform` method combines the `fit` and `transform` methods to calculate the TF-IDF score in a single step.

### Example Usage

```python
corpus = [
    "Apple Apple Banana",
    "Banana Mango Banana",
    "Cherry Cherry Cherry",
    "Grapes Grapes Berries Grapes",
    "Apple Banana Mango",
    "Blueberries Strawberries Apple",
    "Apple Banana Mango",
    "Grapes Grapes Grapes",
    "Blueberries Apple Strawberries",
    "Apple Banana Apple",
    "Cherry Cherry Mango Cherry",
    "Blueberries Strawberries Cherry",
]

def format_matrix(vocab: dict, matrix: np.ndarray) -> pd.DataFrame:
    if len(vocab) == len(matrix[0]):
        terms = list(vocab)
        return pd.DataFrame(
            data=matrix,
            columns=terms
        )
    else:
        raise ValueError("Vocabulary and Result matrix do not match")


tf_idf = TfidfVectorizer()

tfidf_matrix = tf_idf.fit_transform(corpus)

print(f"Vocab: {tf_idf.vocabulary}")
print(f"Document Frequency: {tf_idf.document_frequency}")
print(f"IDF: {tf_idf.inverse_document_frequency}")
print("Result:")
format_matrix(tf_idf.vocabulary, tfidf_matrix)
```

**Output**:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>apple</th>
      <th>banana</th>
      <th>mango</th>
      <th>cherry</th>
      <th>grapes</th>
      <th>berries</th>
      <th>strawberries</th>
      <th>blueberries</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.197225</td>
      <td>1.223775</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>2.447551</td>
      <td>1.386294</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.828314</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.83773</td>
      <td>2.564949</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.098612</td>
      <td>1.223775</td>
      <td>1.386294</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.609438</td>
      <td>1.609438</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.098612</td>
      <td>1.223775</td>
      <td>1.386294</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.83773</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.098612</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.609438</td>
      <td>1.609438</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.197225</td>
      <td>1.223775</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>4.828314</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.609438</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.609438</td>
      <td>1.609438</td>
    </tr>
  </tbody>
</table>
</div>

### Visualization and explanation

```python
import plotly.graph_objects as go

x_values = [df for _, df in tf_idf.document_frequency.items()]
y_values = [idf for _, idf in tf_idf.inverse_document_frequency.items()]

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=x_values,
        y=y_values,
        mode="markers+lines"
    )
)

fig.update_layout(
    title="DF vs IDF",
    xaxis_title="Document Frequency",
    yaxis_title="Inverse Document Frequency"
)

fig.show()
```

The above chart shows how the IDF score changes as per the number of documents in which the term appears. As the number of documents increases, the IDF score decreases. This is because the term becomes less unique as it appears in more documents.

### Similarity search results

```python
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_docs(query: str, corpus: list[str], docs_vector, tf_idf, top_k: int = 3):
    query_vector = tf_idf.transform([query])
    similarity_score = cosine_similarity(query_vector, result)
    ranked_indices = similarity_score.argsort()[0][::-1][:top_k]
    retrieved_docs = [{"doc": corpus[i], "score": round(float(similarity_score[0][i]), 3)} for i in ranked_indices]
    return retrieved_docs

queries = [
    "Blueberries Strawberries",
    "grapes",
    "cherry",
    "banana mango"
]
query = queries[3]
print(f"Query: {query}")
docs = retrieve_docs(query, corpus, result, tf_idf, top_k=5)
docs
```

**Output:**

```python
Query: banana mango

[{'doc': 'Banana Mango Banana', 'score': 0.945},
 {'doc': 'Apple Banana Mango', 'score': 0.86},
 {'doc': 'Apple Banana Mango', 'score': 0.86},
 {'doc': 'Apple Banana Apple', 'score': 0.322},
 {'doc': 'Apple Apple Banana', 'score': 0.322}]
```

The `retrieve_docs` function takes a query, corpus and returns the top-k most similar documents from the corpus based on the cosine similarity between the query and the documents. The function uses the `cosine_similarity` function from `sklearn` to calculate the similarity scores.

## Conclusion

While TF-IDF has it's limitations, it remains a powerful tool for text analysis and information retrieval. It also layed the foundation for more advanced techniques like BM25 and word embeddings. Understanding the intuition, math, and implementation of TF-IDF is essential for anyone working with text data.

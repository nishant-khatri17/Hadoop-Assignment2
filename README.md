# Hadoop Assignment 2 – MapReduce on Wikipedia Dataset

## Overview

This project implements multiple Hadoop MapReduce tasks on a dataset of 10,000 Wikipedia articles. The objective is to analyze large-scale text data, compare MapReduce design patterns, and evaluate performance across different approaches.

The assignment includes:

* Word frequency analysis
* Co-occurrence matrix generation (Pairs and Stripes)
* Local aggregation optimizations
* Document Frequency (DF)
* TF-IDF computation

---

## Technologies Used

* Apache Hadoop (Local Mode)
* Java
* HDFS
* MapReduce Programming Model
* OpenNLP (Porter Stemmer)

---

## Dataset

* Wikipedia Articles Dataset (10,000 documents)
* Each document is processed as a single input record using a custom input format

---

## Project Structure

```
HADOOP/
├── problem1/
│   ├── src/              # Java source files
│   ├── classes/          # Compiled classes
│   ├── *.jar             # Executable Hadoop jobs
│   ├── stopwords.txt     # Stopword list
│   ├── top50words.txt    # Top 50 frequent words
│   └── top50words_only.txt
│
├── problem2/
│   ├── src/
│   ├── classes/
│   ├── *.jar
│   ├── stopwords.txt
│   ├── top100df.txt
│   └── top100terms.txt
│
├── outputs/
│   ├── problem1/         # Full outputs for co-occurrence tasks
│   └── problem2/         # Document Frequency and TF-IDF outputs
│
├── report/               # Final report
└── README.md
```

---

## How to Run

### Compile Java Code

```
javac -cp "$(hadoop classpath)" -d . src/*.java
```

### Create JAR

```
jar cf program.jar *.class
```

### Run Hadoop Job

```
hadoop jar program.jar MainClass input output
```

---

## Problem Breakdown

### Problem 1: Co-occurring Word Matrix

#### Top 50 Words

* Word count using MapReduce
* Stopwords removed using distributed cache
* Output sorted to extract top 50 words

#### Pairs Approach

* Emits (word1, word2) pairs
* Generates a large number of intermediate keys
* Runtime increases with window size d

#### Stripes Approach

* Emits (word, map of neighbors)
* Reduces number of emitted keys
* Higher memory usage due to MapWritable

#### Pairs vs Stripes

| Aspect                | Pairs  | Stripes |
| --------------------- | ------ | ------- |
| Emissions             | High   | Low     |
| Shuffle Cost          | High   | Lower   |
| Memory Usage          | Low    | High    |
| Performance (small d) | Faster | Slower  |

#### Local Aggregation

* Map-Function level aggregation
* Map-Class level aggregation
  These optimizations reduce shuffle size and improve performance, especially for Stripes.

---

### Problem 2: Document Indexing

#### Document Frequency (DF)

* Counts the number of documents containing a term
* Uses a HashSet to ensure each term is counted once per document
* Applies Porter Stemming

#### TF-IDF Computation

TF-IDF = TF × log(10000 / DF + 1)

* TF: term frequency in a document
* DF: number of documents containing the term

Higher TF-IDF values indicate terms that are important within a document but relatively rare across the dataset.

---

## Key Results

* Extracted top 50 most frequent words
* Generated co-occurrence matrices using Pairs and Stripes
* Compared performance across different window sizes
* Implemented local aggregation optimizations
* Computed TF-IDF scores for documents

---

## Runtime Summary

| Task               | Runtime      |
| ------------------ | ------------ |
| Pairs (d = 1 to 4) | 5–10 minutes |
| Stripes            | 8–11 minutes |
| Document Frequency | ~8 minutes   |
| TF-IDF             | ~6 minutes   |

All experiments were conducted on a single-node Hadoop setup.

---

## Observations

* Pairs performs better for smaller window sizes
* Stripes becomes more efficient as the window size increases
* Local aggregation significantly reduces shuffle overhead
* MapWritable introduces serialization overhead in Stripes
* TF-IDF effectively identifies document-specific important terms

---

## Execution Environment

* Hadoop Local Mode (single-node setup)
* macOS system
* No YARN cluster used

---

## Notes

* Full outputs are included in the outputs directory
* Dataset is not included due to size constraints
* Results are reproducible using the provided code

---

## Authors

* Nishant Khatri
* Siddharth Goswami
* Arshbir Singh Dang
* Manav Jindal

---

## Report

Detailed explanation is available in:
report/hadoop.pdf

---

## Conclusion

This project demonstrates practical implementation of Hadoop MapReduce and highlights the trade-offs between different approaches such as Pairs and Stripes. It also shows how local aggregation improves performance and how TF-IDF can be applied to extract meaningful insights from large text datasets.

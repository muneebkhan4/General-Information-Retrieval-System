import os
import re
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

# read documents from text files
corpus_path = r"Corpus 2"
documents = []
for file_name in os.listdir(corpus_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(corpus_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            document = f.read()
            documents.append(document)

# create list of stop words
stop_words = set()
with open('Stopwords.txt', 'r') as f:
    stop_words = set(f.read().splitlines())
# print(stop_words)

# create stemmer object
stemmer = PorterStemmer()

# preprocess documents
processed_documents = []
for document in documents:
    # tokenize into words
    words = nltk.word_tokenize(document)
    # convert to lowercase
    words = [word.lower() for word in words]
    # remove stop words
    words = [word for word in words if word not in stop_words]
    # stem words
    words = [stemmer.stem(word) for word in words]
    processed_documents.append(words)

# create vocabulary of words
vocabulary = list(set([word for document in processed_documents for word in document]))

# print(vocabulary)

# calculate tf-idf values
tfidf_matrix = []
for document in processed_documents:
    tfidf_vector = []
    for word in vocabulary:
        tf = document.count(word) / len(document)
        idf = math.log(len(processed_documents) / (1 + sum([1 for d in processed_documents if word in d])))
        tfidf_vector.append(tf * idf)
    tfidf_matrix.append(tfidf_vector)

# calculate query tf-idf vector
query = input("Enter query: ")
query_words = nltk.word_tokenize(query)
query_words = [word.lower() for word in query_words if word.lower() not in stop_words]
query_words = [stemmer.stem(word) for word in query_words]

print(query_words)

query_tfidf_vector = []
for word in vocabulary:
    tf = query_words.count(word) / len(query_words)
    idf = math.log(len(processed_documents) / (1 + sum([1 for d in processed_documents if word in d])))
    query_tfidf_vector.append(tf * idf)

# calculate cosine similarity scores
document_scores_cosine = []
for i, document in enumerate(tfidf_matrix):
    dot_product = sum([a * b for a, b in zip(document, query_tfidf_vector)])
    document_magnitude = math.sqrt(sum([a ** 2 for a in document]))
    query_magnitude = math.sqrt(sum([a ** 2 for a in query_tfidf_vector]))
    document_score = dot_product / (document_magnitude * query_magnitude)
    document_scores_cosine.append((i, document_score))

# print document scores based on cosine similarity in descending order
document_scores_cosine = sorted(document_scores_cosine, key=lambda x: x[1], reverse=True)
print("\nRanking based on Cosine Similarity Score:")
for i, score in enumerate(document_scores_cosine):
    document_index = score[0]
    document_score = score[1]
    print(f"{i+1}. Document {document_index+1}: {document_score:.4f}")
    if i == 9:
        break

# calculate tf-idf scores
document_scores_tfidf = []
for i, document in enumerate(tfidf_matrix):
    tfidf_vector = []
    for word in vocabulary:
        if word in vocabulary:
            tf = document[vocabulary.index(word)] / sum(document)
            idf = math.log(len(processed_documents) / (1 + sum([1 for d in processed_documents if word in d])))
            tfidf_vector.append(tf * idf)
        else:
            tfidf_vector.append(0)

    document_score = sum([tfidf_vector[j] for j, word in enumerate(vocabulary) if word in query_words])
    document_scores_tfidf.append((i, document_score))



# print document scores based on tf-idf score in descending order
document_scores_tfidf = sorted(document_scores_tfidf, key=lambda x: x[1], reverse=True)
print("\n\nRanking based on TF-IDF Score:")

for i, score in enumerate(document_scores_tfidf):
    document_index = score[0]
    document_score = score[1]
    print(f"{i+1}. Document {document_index+1}: {document_score:.4f}")
    if i == 9:
        break
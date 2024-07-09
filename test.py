from sklearn.feature_extraction.text import TfidfVectorizer

# Example preprocessed text data
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit-transform the corpus to TF-IDF vectors
tfidf_vectors = vectorizer.fit_transform(corpus)

# Get feature names (words/tokens)
feature_names = vectorizer.get_feature_names_out()

# Print TF-IDF vectors and feature names
print(tfidf_vectors.toarray())
print(feature_names)

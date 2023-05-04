import argparse
import os
import sys
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == '__main__':
    
    # Take the input from users using argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", type=str, required=True, help="Enter the path of the city pdf file")
    parser.add_argument("--summarize", action='store_true')
    parser.add_argument("--keywords", action='store_true')
    
    # Retrieve the arguments
    args = parser.parse_args()
    
    document_Path = args.document
    summarize = args.summarize
    keywords = args.keywords
    
    print(document_Path, summarize, keywords)
    
def predict():    
    filePath = 'smartcity/NV Las Vegas.pdf'
    reader = PdfReader(filePath)
    extracted_text = []

    # Iterate through each page in the PDF file
    for page_num in range(len(reader.pages)):

        page = reader.pages[page_num]
        # Extract the text from the page
        text = page.extract_text()

        # Add the extracted text to the list
        extracted_text.append(text)

    #extracted_text = normalize_corpus(extracted_text)

    # Concatenate the strings in the list into a single string
    full_text = ' '.join(extracted_text)

    #full_text = normalize_corpus(full_text)

    kmeans = joblib.load('model.pkl')
    X_new = vectorizer.transform([full_text])
    cluster_label = kmeans.predict(X_new)

    print(cluster_label)
    
    
def train a model():
    from sklearn.naive_bayes import MultinomialNB

    # Select the input variables as X
    X = dataFrame['raw text']

    # Select the target variable as Y
    Y = dataFrame['clusterid']

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)

    # Train the model
    model = MultinomialNB()
    model.fit(X, Y)

    # Save the model to disk
    dump(model, 'model.pkl')
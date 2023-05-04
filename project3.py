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
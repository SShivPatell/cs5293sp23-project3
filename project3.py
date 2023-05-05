import argparse
from joblib import dump, load
import os
import sys
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import nltk
import spacy
import unicodedata
import re
from nltk.corpus import wordnet
import collections
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
import pandas as pd


# used the code given in text_normalizer.py
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    if bool(soup.find()):
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    else:
        stripped_text = text
    return stripped_text

def simple_porter_stemming(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
            
    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]|\[|\]' if not remove_digits else r'[^a-zA-Z\s]|\[|\]'
    text = re.sub(pattern, '', text)
    return text


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

# Removes the states name
def remove_States(text):
    pattern = r'Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\sHampshire|New\sJersey|New\sMexico|New\sYork|North\sCarolina|North\sDakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\sIsland|South\sCarolina|South\sDakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West\sVirginia|Wisconsin|Wyoming|AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY'                        
    text = re.sub(pattern, '', text)
    return text

# Removes the common words which I think are not that important in the pdf files
def remove_CommonWords(text):
    pattern = r"U\.S\. Department|use|traffic|datum|system|smartcity|smart|city|page|section|element|concept|appendix|transportation|(\s(a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z)\s)|(\s\w\w\s)"
    text = re.sub(pattern, '', text)
    return text

# Removes the city names from the pdf files
def remove_Cities(text):
    pattern = r'Anchorage|Birmingham|Montgomery|Scottsdale|Tucson|Chula Vista|Fremont|Fresno|Long Beach|Moreno Valley|Oakland|Oceanside|Riverside|Sacramento|San Jose_0|NewHaven|DC_0|Jacksonville|Miami|Orlando|St. Petersburg|Tallahassee|Tampa|Atlanta|Brookhaven|Des Moines|Indianapolis|Louisville|Baton Rogue|New Orleans|Shreveport|Boston|Baltimore|Detroit|Port Huron and Marysville|Minneapolis St Paul|St. Louis|Charlotte|Greensboro|Raleigh|Lincoln|Omaha|Jersey City|Newark|Las Vegas|Reno|Albany Troy Schenectady Saratoga Springs|Buffalo|Mt Vernon Yonkers New Rochelle|Rochester|Akron|Canton|Cleveland|Toledo|Oklahoma City|Tulsa|Providence|Greenville|Chattanooga|Memphis|Nashville|Lubbock|Newport News|Norfolk|Richmond|Virginia Beach|Seattle|Spokane|Madison'                        
    text = re.sub(pattern, '', text)
    return text

def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_stemming=True, text_lemmatization=True, 
                     special_char_removal=True, remove_digits=True,
                     stopword_removal=True, removeStates = True, removeCities = True,
                     removeCommonWords = True, stopwords=stopword_list):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:

        # strip HTML
        if html_stripping:
            doc = strip_html_tags(doc)

        # remove extra newlines
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))

        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)

        # expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)

        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)

        # stem text
        if text_stemming and not text_lemmatization:
        	doc = simple_porter_stemming(doc)

        # remove special characters and\or digits    
        if special_char_removal:
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)  

        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)

         # remove states
        if removeStates:
            doc = remove_States(doc)
            
        # remove cities
        if removeCities:
            doc = remove_Cities(doc)
            
         # lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # remove the common words
        if removeCommonWords:
            doc = remove_CommonWords(doc)
            
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case, stopwords=stopwords)
            
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus

# Read the data of the pdf and returns it as a dataframe
def readPdf(fileName, document_Path):
    
    reader = PdfReader(document_Path)
    extracted_text = []
    dataFrame = pd.DataFrame(columns=['city', 'raw text'])

    # Iterate through each page in the PDF file
    for page_num in range(len(reader.pages)):

        page = reader.pages[page_num]
        # Extract the text from the page
        text = page.extract_text()
        
        # Add the extracted text to the list
        extracted_text.append(text)
    data_dict = {'city': fileName, 'raw text': extracted_text}
    dataFrame = pd.concat([dataFrame, pd.DataFrame(data_dict)], ignore_index=True)
    dataFrame = dataFrame.groupby("city")["raw text"].apply(lambda x: " ".join(x)).reset_index()
    return dataFrame

# Predicts the clusterid
def predict(newData):
    # Normalize the new data
    newData['clean text'] = normalize_corpus(newData['raw text'])

    new_text = newData['clean text']
    
     #Load the model and vectorizer
    kmeans = load('model.pkl')
    vectorizer = load('vectorizer.pkl')

    # Transform the new data using the fitted vectorizer
    new_tfidf = vectorizer.transform(new_text)

    # Predict the clusters for the new data using the fitted KMeans model
    predicted_clusters = kmeans.predict(new_tfidf)

    return predicted_clusters, newData

# Used to update the smartcity_predict.tsv file
def writeToTsv(dataFrame):

    # Set the file name
    file_name = 'smartcity_predict.tsv'

    # Check if the file exists
    if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
        # File exists and is not empty, append row to file
        df = pd.read_csv(file_name, sep='\t', escapechar='\\')
        df = pd.concat([df, dataFrame], ignore_index=True)
        df.to_csv(file_name, sep='\t', escapechar='\\', index=False)
    else:
        # File does not exist or is empty, create new file and add header row and data row
        dataFrame.to_csv('smartcity_predict.tsv', sep = '\t', escapechar = '\\', index = False)


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
    
    # Get the name of the city
    fileName = os.path.splitext(os.path.basename(document_Path))[0]
    data = readPdf(fileName, document_Path)
    clusterid, dataFrame = predict(data)
    
    # Write to the smartcity_predict.tsv file
    dataFrame['clusterid'] = clusterid
    # Write to smartcity_predict.tsv
    writeToTsv(dataFrame)
    
    # print the output
    print(f"[{fileName}]" + " clusterid: " + str(clusterid[0]))
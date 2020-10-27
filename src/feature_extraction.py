from sklearn.feature_extraction import text
from nltk import word_tokenize

features_dict = {
    "TfidVectorizer" : text.TfidfVectorizer(tokenizer = word_tokenize,
                                     token_pattern = None,
                                     stop_words = []),
    "CountVectorizer" : text.CountVectorizer(tokenizer = word_tokenize,
                                     token_pattern = None,
                                     stop_words = [])
}
#please install contractions
import numpy as np
import re
import nltk
import contractions
import string
import pandas as pd

import numpy as np
import random
import time
import re
import unidecode
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from gensim.test.utils import get_tmpfile
from nltk.corpus import stopwords
import nltk
import spacy
import en_core_web_sm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import tensorflow as tf
import matplotlib.pyplot as plt

nlp = en_core_web_sm.load()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
porter=PorterStemmer()
lancaster=LancasterStemmer()
REGEX = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
         '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

NEWLINE_REGEX = ('[\n\r\t]')


def replace_contractions(text) :
    # print(text)
    """Replace contractions in string of text"""
    return contractions.fix ( text )


def replace_ne(text: str, nlp) -> str :
    '''
    This function extracts the name entities, and then replace them with the ne labels.
    :param text:
    :type text:
    :return:
    :rtype:
    '''
    doc = nlp ( text )
    for ent in doc.ents :
        text = text.replace ( ent.text, ent.label_ )
    return text


def textClean(text) :
    """
    Get rid of the non-letter and non-number characters
    """
    # text = re.sub(REGEX, " ", text)
    # text = re.sub(NEWLINE_REGEX, " ", text)
    # text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
    text = re.sub ( '[^A-Za-z0-9]+', ' ', text )
    re.sub ( r'@\w+', ' ', text )

    # text = unidecode.unidecode(text)

    text = text.lower ().split ()
    stops = set ( stopwords.words ( "english" ) )
    text = [t for t in text if t.isalpha ()]
    lemmatizer = WordNetLemmatizer ()
    text = [lemmatizer.lemmatize ( t ) for t in text]

    text = [w for w in text if not w in stops]
    text = " ".join ( text )
    return (text)


def stemSentence(sentence) :
    token_words = word_tokenize ( sentence )

    stem_sentence = []
    for word in token_words :
        stem_sentence.append ( lancaster.stem ( word ) )
        stem_sentence.append ( " " )
    return "".join ( stem_sentence )


def cleanup(text) :
    text = replace_contractions ( text )
    text = textClean ( text )
    # text = text.translate(str.maketrans("", "", string.punctuation))
    # text = replace_ne(text,nlp)

    text = stemSentence ( text )
    return text


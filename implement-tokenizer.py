# The following required libraries are needed.
# Install the libraries from the terminal command ... Remove install commands before running implemented-tokenizer.py 

pip3 install nltk
pip3 install transformers
pip3 install sentencepiece
pip3 install spacy
pip3 install numpy
pip3 install numpy scikit-learn
pip3 install torch
pip3 install torchtext
python3 -m spacy download en_core_web_sm
python3 -m spacy download de_core_news_sm


# Importing required libraries
import nltk
nltk.download("punkt")
nltk.download('punkt_tab')
import spacy
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams
from transformers import BertTokenizer
from transformers import XLNetTokenizer

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


# Word-based tokenizer
# As the name suggests, this is the splitting of text based on words.
# There are different rules for word-based tokenizers, such as splitting
# on spaces or splitting on punctuation. Each option assigns a specific
# ID to the split word.

# nltk
text = "This is a sample sentence for word tokenization."
tokens = word_tokenize(text)
print(tokens)

# Output:
# ['This', 'is', 'a', 'sample', 'sentence', 'for', 'word', 'tokenization', '.']

# General libraries like nltk and spaCy often split words like 'don't' and 'couldn't,'
# which are contractions, into different individual words. There's no universal rule,
# and each library has its own tokenization rules for word-based tokenizers. However,
# the general guideline is to preserve the input format after tokenization to match 
# how the model was trained.
# This showcases word_tokenize from nltk library

text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."
tokens = word_tokenize(text)
print(tokens)

# Output:
# ['I', 'could', "n't", 'help', 'the', 'dog', '.', 'Ca', "n't", 'you', 'do', 'it', '?', 'Do', "n't", 'be', 'afraid', 'if', 'you', 'are', '.']


# This showcases the use of the 'spaCy' tokenizer with torchtext's get_tokenizer function

text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# Making a list of the tokens and priting the list
token_list = [token.text for token in doc]
print("Tokens:", token_list)

# Showing token details
for token in doc:
    print(token.text, token.pos_, token.dep_)

# Output:
# Tokens: ['I', 'could', "n't", 'help', 'the', 'dog', '.', 'Ca', "n't", 'you', 'do', 'it', '?', 'Do', "n't", 'be', 'afraid', 'if', 'you', 'are', '.']
# I PRON nsubj
# could AUX aux
# n't PART neg
# help VERB ROOT
# the DET det
# dog NOUN dobj
# . PUNCT punct
# Ca AUX aux
# n't PART neg
# you PRON nsubj
# do VERB ROOT
# it PRON dobj
# ? PUNCT punct
# Do AUX aux
# n't PART neg
# be AUX ROOT
# afraid ADJ acomp
# if SCONJ mark
# you PRON nsubj
# are AUX advcl
# . PUNCT punct


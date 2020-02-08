import operator
import logging
import pickle
from functools import reduce
from itertools import chain

from pathos.multiprocessing import ProcessingPool as Pool
from nltk.util import ngrams
import numpy as np
import csv
import gzip

import unicodedata
import time
from bs4 import BeautifulSoup
from pycontractions import Contractions
import gensim.downloader as api


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp<=126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def run_split_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if is_punctuation(char):
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]

def run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))

def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        if is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def preprocess_line(text):
    text = convert_to_unicode(text)
    text = clean_text(text)
    tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in tokens:
        token = token.lower()
        token = run_strip_accents(token)
        split_tokens.extend(run_split_on_punc(token))
    return " ".join(split_tokens)

class L3wTransformer:
    """
    Parameters
    ----------
    max_ngrams : The upper bound of the top n most frequent ngrams to be used. If None use all containing ngrams.
    ngram_size : The size of the ngrams.
    lower : Should the ngrams be treated as lower char ngrams.
    split_char : Delimeter for splitting strings into a list of words.
    """

    def __init__(self, max_ngrams=50000, ngram_size=3, lower=True, split_char=None, parallelize=False):
        self.ngram_size = ngram_size
        self.lower = lower
        self.split_char = split_char
        self.max_ngrams = max_ngrams
        self.indexed_lookup_table = {}
        self.parallelize = parallelize
        self.tmp = 0
        self.tmp1 = 0

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            dump_dict = pickle.load(f)

        l3wt = L3wTransformer(
            max_ngrams=dump_dict['max_ngrams'],
            ngram_size=dump_dict['ngram_size'],
            lower=dump_dict['lower'],
            split_char=dump_dict['split_char'],
            parallelize=dump_dict['parallelize']
        )

        l3wt.indexed_lookup_table = dump_dict['indexed_lookup_table']
        return l3wt

    def save(self, path):
        dump_dict = {
            'ngram_size': self.ngram_size,
            'lower': self.lower,
            'split_char': self.split_char,
            'max_ngrams': self.max_ngrams,
            'indexed_lookup_table': self.indexed_lookup_table,
            'parallelize': self.parallelize
        }
        with open(path, 'wb') as f:
            pickle.dump(dump_dict, f)

    def word_to_ngrams(self, word):
        """Returns a list of all n-gram possibilities of the given word."""
        if self.lower:
            word = word.lower()
        word = '<' + word + '>'
        return list(map(lambda x: ''.join(x), list(ngrams(word, self.ngram_size))))

    def scan_paragraphs(self, offsets):
        """Creates a lookup table from the given paragraphs, containing all
        n-gram frequencies."""
        lookup_table = {} 
        with open("msmarco-docs.tsv", encoding="utf8") as f:
            def fill_lookup_table(lookup_table, offset): 
                f.seek(offset)
                line = f.readline().rstrip().split('\t')[2:]
                line = ''.join(line)
                line = preprocess_line(line)
                words = line.split()
                for w in words:
                    ngrams_w = self.word_to_ngrams(w)
                    for n in ngrams_w:
                        if n in lookup_table:
                            lookup_table[n] = lookup_table[n] + 1
                        else:
                            lookup_table[n] = 1
                return lookup_table
    
            lookup_table = reduce(fill_lookup_table, offsets, {})
        return lookup_table


    def fit_on_texts(self, texts):
        """Convenient method for creating a indexed lookup table,
        necessary to transform text into a integer sequence. Always call this before
        texts_to_sequences method to get results.
        Returns the indexed lookup table."""
        if not texts:
            return []

        lookup_table = self.scan_paragraphs(texts)
        if not self.max_ngrams:
            self.max_ngrams = len(lookup_table)

        sorted_lookup = sorted(lookup_table.items(),
                               key=operator.itemgetter(1), reverse=True)
        indexed_lookup_table = dict(
            zip(list(zip(*sorted_lookup[:self.max_ngrams]))[0],  # get only the max_ngrams frequent tri grams
                list(range(1, self.max_ngrams + 1)))
        )

        # self.indexed_lookup_table = indexed_lookup_table
        # return self.indexed_lookup_table
    
with gzip.open("msmarco-docs-lookup.tsv.gz", 'rt', encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    offsets = []
    t1 = time.time()
    
    """
    Training on all the offsets one by one may take like 15 hours on the laptop
    for processing MS-MARCO documents corpus of size 22 GB.
    It is advisable to break the offsets list into 6 equal sized lists, and then 
    build just the lookup_tables (not the indexed_lookup_table), for all of 6
    lists in parallel manner by processing on different CPU cores, and then merge 
    the resulting lookup_tables. Finally, the indexed lookup table can be built
    easily from this merged lookup table. 
    """
    
    for [docid, _, offset] in tsvreader:
        offsets.append(int(offset))
    a = L3wTransformer(max_ngrams=30000)
    a.fit_on_texts(offsets)
    a.save("hash_table.pickle")
    print(time.time()-t1)





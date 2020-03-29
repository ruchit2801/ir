# ir

The model CDSSM is a IR model to retrive documents in web searching, originally published by microsoft, and can be found [here](https://www.microsoft.com/en-us/research/publication/a-latent-semantic-model-with-convolutional-pooling-structure-for-information-retrieval/).

## Idea
Use structure similar to CDSSM model, but for training 300 dimensional word embedings, I used different approach.<br>

<b>(1) </b> I used [sentencepiece](https://github.com/google/sentencepiece) tokenizer first. For which I trained a sentencepiece model on the MSMARCO corpus. I used the vocabulary size of 30000 for this task. Since the corpus size is 22.9 GB, we have to either sample some number of sentences out of corpus(e.g. randomly sampling 1 million sentences. We'll have memory issues if we want to use whole corpus for model training using sentencepiece), or, we can provide the corpus in the form of a tsv file in which we have word frequencies of every word occuring in the corpus. MS-MARCO corpus contained more than 228 million sentences, so the first appoach (randomly sampling some number of sentences) gives relatiely less coverage on the corpus. That's why I decided to go for second approach. I first created a tsv file of word frequencies and then fed it to sentencepiece model for the training using following command. 

```
spm_train --input=word-freq.tsv --model_prefix=model --vocab_size=30000 --character_coverage=1.0 --input_format=tsv
```

<b>(2) </b> This produced a nice tokenizer for me, using unigram language model algorithm. The next task was to train from the scratch word embeddings of these 30000 different tokens. We decided to use `word2vec` algorithm for training embeddings. I used gensim word2vec model for this. The script to train embeddings is as follows. 

```python
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("model.model")

class Corpus():
    """Iterate over sentences from the corpus."""
    def __init__(self):
        self.files = [
            "corpus.txt"
        ]

    def __iter__(self):
        for fname in self.files:
            for line in open(fname):
                words = sp.EncodeAsPieces(line)
                yield words

sentences = Corpus()

model = Word2Vec(sentences, size=300, window=5, min_count=1, workers=8, sg=1, negative=10)
model.save("word2vec.model")
```
A piece of advice from my side would be not to use `negative=10`(I used this value, but I advice not to) for such big corpus. It almost doubles the training time. Since we have pretty large corpus, using negative=10 will not give significant performance improvement as compated to investing that CPU time on some other task. Use the default value of 5 for negative sampling.  I provide here some interesting results after training embeddings. I explored following clusters from trained embeddings. 

* Surnames : Singh, Patel, Sharma, Kumar, Raj, Narayan 
* company Amazon : kindle, ebay, hulu, roku, walmart, ebook
* company Microsoft : sdk, windows, msdn, desktop, software, xp, ibm, server
* gods : shiva, vishnu, krishna, brahma, dharma, buddha, hindu, lakshmi

And many more such clusters.

<b>I've added sentencepiece model. Due to limited internet, I can not add trained word2vec model right now. Will do that soon. </b>

## Next UP : 
 
* Train the CDSSM like model for document retrieval. Instead of training embeddings in CDSSM model, use above trained embeddnigs for training. So, training will involve first tokenising a query and document using trained sentencepiece model, then obtain their word embeddings from trained word2vec model. After that, either use max pooling or averaging over query/document similar to that used in original CDSSM paper. 

# Acknowledgements : 
* I thank my mentor [Parth Gupta](https://pgupta.gitlab.io/), for his guidance and valuable feedbacks throughout this project.
* I also thank [Prof. Prasenjit Majumder](https://www.daiict.ac.in/profile/prasenjit-majumder/) for supervising and guiding this project.



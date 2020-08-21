# Neural Information retrieval model : 

This project implements the exact replica of Convolutional Deep Structured Semantic Model (CDSSM) or Convolutinal Latent Semantic Model (CLSM) introduced by Microsoft Research in 2014.
[This](https://www.microsoft.com/en-us/research/publication/a-latent-semantic-model-with-convolutional-pooling-structure-for-information-retrieval/) is link to original work.

The model has been implemented in PyTorch, and is ready to use model. To use this model, one needs to keep ready training and test data in the following format. 

(1) `Queries` : Each query in the training/validation data has to be converted in to sequence of `30k` dimensional tensors. This can be done using the word hashing mechanism 
introduced in the paper. Store these sequences in the form of dictionary where keys will be query ids and values would be list of lists. In this list, each list
represents the words of the input query based on letter-trigram based word hashing. (Each list stores indices of letter trigrams of word in the query). See the notebook 
for the example of required format. 

(2) `Documents` : Similar to queries, we have exact same format for documents. 

(3) `qrels data` : Training data in pairwise format. Each pair in the training data has the form (Q, D) of query and its relevant document.

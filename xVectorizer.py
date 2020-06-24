import pandas as pd
from typing import List
import scipy

import xCustomVectorizer as xCV

class xVectorizer(object):
    def __init__(self, data_corpora = None):

        #to be replaced
        self.__options = { "test": 10 }

        #a holder of data
        self.__data_corpora = data_corpora

        #sparse document term matrix (DTM)
        self.__data_matrix = None

        #matrix in array format
        self.__data_array = None

        #matrix in data_frame format
        self.__data_frame = None

        #names of data
        self.__data_names = None

        #names of features
        self.__feature_names = None

        #frequencies
        self.__frequencies = None

        #occurences
        self.__occurences = None

        #sparsity
        self.__sparsity = None

        #NZC
        self.__non_zero_counts = None

        #vocabulary
        self.__vocabulary = None

        #stopwords
        self.__stopwords = None
        
        #vectorizer class
        self.__vectorizer = None

        #fit and transform
        self.__applied = False
    
    def __str__(self):
        return self.data_frame.to_string()

    def __repr__(self):
        return "{self.__class__.__name__}\n{self.data_frame.to_string()}"

    @property
    def data_corpora(self):
        return self.__data_corpora

    @data_corpora.setter
    def data_corpora(self, data = None):
        self.__data_corpora = data
    
    @property
    def data_matrix(self) -> scipy.sparse.csr.csr_matrix:
        return self.__data_matrix

    @property
    def data_term_matrix(self) -> scipy.sparse.csr.csr_matrix:
        return self.__data_matrix

    @property
    def applied(self):
        return self.__applied

    @property
    def data_frame(self):
        """
        - transform the sparse matrix into panda data_frame 
        - created uppon request
        """
        if not self.__data_frame:
            self.set_data_frame()
        return self.__data_frame

    @property
    def data_array(self):
        """
        - transform the sparse matrix into array
        - created uppon request
        """
        if not self.__data_array:
            self.set_data_array()
        return self.__data_array

    @property
    def data_names(self):
        """
        - create an index for each document
        - can serve as row names
        - created uppon request
        """
        if not self.__data_names:
            self. set_data_names()
        return self.__data_names

    @property
    def feature_names(self):
        """
        - obtain feature names from vectorizer
        - created uppon request
        """
        if not self.__feature_names:
             self.set_feature_names()
        return self.__feature_names

    @property
    def frequencies(self):
        """
        - obtain frequencies from vectorizer
        - created uppon request
        """
        if not self.__frequencies:
             self.set_frequencies()
        return self.__frequencies

    @property
    def data_frame_tf(self):
        return self.frequencies
    
    @property
    def occurences(self):
        """
        - obtain frequencies from vectorizer
        - created uppon request
        """
        if not self.__occurences:
             self.set_occurences()
        return self.__occurences
    
    @property
    def sparsity(self):
        """
        - sparsity of sparse matrix
        - created uppon request
        """
        if not self.__sparsity:
             self.set_sparsity()
        return self.__sparsity

    @property
    def non_zero_counts(self):
        """
        - non-zero counts of sparse matrix
        - created uppon request
        """
        if not self.__non_zero_counts:
             self.set_non_zero_counts()

        return self.__non_zero_counts

    @property
    def vocabulary(self):
        """
        - vocabulary of vectorizer
        - created uppon request
        """
        if not self.__vocabulary:
             self.set_vocabulary()
        return self.__vocabulary

    @property
    def stopwords(self):
        """
        - stopwaords of vectorizer
        - created uppon request
        """
        if not self.__stopwords:
             self.set_stopwords()
        return self.__stopwords

    def set_feature_names(self):
        self.__feature_names = self.__vectorizer.get_feature_names()
        
    def set_data_names(self, override=False):
        if override:
            self.__data_names = ['doc{:d}'.format(idx) for idx in enumerate(self.__data_corpora.names)]
        else:
            self.__data_names = self.__data_corpora.names

    def set_data_array(self):
        if self.__data_matrix.size:
            self.__data_array = self.__data_matrix.toarray()

    def set_data_frame(self):
        """
        transform a sparse matrix to a pandas data frame
        """
        self.__data_frame = pd.DataFrame(data = self.data_array,
                                         index = self.data_names,
                                         columns = self.feature_names)
    
    def set_frequencies(self):
        if self.__data_matrix.size:
            if len(self.feature_names) !=  len(sum(self.__data_matrix).toarray()[0]):
                print("frequencies: feature names and term matrix have incompatible sizes")
                return None
            self.__frequencies = pd.DataFrame( {
                                    "feature":self.feature_names,
                                    "frequency":sum(self.__data_matrix).toarray()[0],
            }
            )
            #self.__frequencies.set_index('feature', inplace=True)

    def set_occurences(self):
        if self.__data_matrix.size:
            if len(self.feature_names) !=  len(sum(self.__data_matrix).toarray()[0]):
                print("frequencies: feature names and term matrix have incompatible sizes")
                return None
        self.__occurences = pd.DataFrame({'feature':
                                          self.feature_names,
                                          'occurrences':
                                          np.asarray(self.__data_matrix.sum(axis=0)).ravel().tolist()})
        self.__occurences.sort_values(by='occurrences', ascending=False, inplace = True)
                                         
    def set_sparity(self):
        if self.__data_matrix.size:
            self.__sparsity = self.__data_matrix.nnz /\
                (self.__data_matrix.shape[0] * self.__data_matrix.shape[1]) 

    def set_non_zero_counts(self):
        if self.__data_matrix.size:
            self._non_zero_counts = self.__data_matrix.nnz
            
    def set_vocabulary(self):
        if self.__vectorizer:
            self.__vocabulary = self.__vectorizer.vocabulary_

    def set_stopwords(self):
        if self.__vectorizer:
            self.__stopwords = self.__vectorizer.stop_words_
            
    def set_vectorizer(self):
        self.__vectorizer = xCV.custom_vectorizer


    def fit_transform(self):
        """ 
        - Learns the vocabulary dictionary  
        - Creates a document-term matrix (DTM) of shape (n_samples, n_features)
        - instantiates and caches the vectorizer 
        """
        if not self.__vectorizer:
            self.set_vectorizer()

        self.__data_matrix = self.__vectorizer.fit_transform(raw_documents = self.__data_corpora.texts)


    def transform(self, documents):
        """
        Transform documents to document-term matrix.
        """
        return self.__vectorizer.transform(raw_documents = documents)
                                   
    def apply(self):
        """
        user callable method 
        """
        self.fit_transform()
        self.__applied = True

    

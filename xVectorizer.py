from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from typing import List

class xVectorizer(object):
    def __init__(self, data = None):

        #to be replaced
        self.__options = { "test": 10 }

        #a holder of data
        self.__data = data

        #spare term matrix
        self.__data_matrix = None

        #matrix in array format
        self.__data_array = None

        #matrix in data_frame format
        self.__data_frame = None

        #names of data
        self.__data_names = None

        #names of features
        self.__feature_names = None
        
        #vectorizer class
        self.__vectorizer = None


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

    def set_feature_names(self):
        self.__feature_names = self.__vectorizer.get_feature_names()
        
    def set_data_names(self, override=False):
        if override:
            self.__data_names = ['doc{:d}'.format(idx) for idx in enumerate(self.__data.names)]
        else:
            self.__data_names = self.__data.names

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
            
    def set_vectorizer(self):
        self.__vectorizer = CountVectorizer()

    def fit_transform(self):
        """ 
        Learns the vocabulary dictionary and 
        returns document-term matrix 
        of shape (n_samples, n_features)
        """
        print("HELLO", self.__data.texts)
        self.__data_matrix = self.__vectorizer.fit_transform(raw_documents = self.__data.texts)
        print("HEY", self.__data_matrix)

    def apply(self):
        self.set_vectorizer()
        self.fit_transform()

        print('CIAO')
        print(self.data_frame)

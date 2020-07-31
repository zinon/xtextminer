import xCustomTransformer as xCT
import pandas as pd
import numpy as np
import scipy


class xTransformer:
    def __init__(self, vectorizer = None):

        #NOTE: keep it central?
        self.__vectorizer = vectorizer
        
        #matrix of term/token counts
        self.__data_term_matrix = self.__vectorizer.data_term_matrix

        #NOTE: keep it central?
        #names of features
        self.__feature_names = self.__vectorizer.feature_names

        #transformer instance
        self.__transformer = None

        #ndarray array of shape
        self.__array_shape = None

        # idf matrix
        self.__data_frame_idf = None

        #apply process
        self.__applied = False

        ##########################
        #tf-idf computational part
        ##########################

        #NOTE: keep it central?
        # list of documents for which tf-idf will be created
        self.__docs = None

        #generated word of counts
        self.__docs_matrix_word_count = None

        #generated matrix of tf-idf weights
        self.__docs_matrix_tf_idf = None

        #generated dataframe of tf-idf weights
        self.__docs_frame_tf_idf = None

        #caching flag for repeated tf-idf calls in the same doc
        self.__docs_frame_tf_idf_set = False
        
    @property
    def data_term_matrix(self) -> scipy.sparse.csr.csr_matrix:
        return self.__data_term_matrix

    @data_term_matrix.setter
    def data_term_matrix(self, matrix:scipy.sparse.csr.csr_matrix = None):
        self.__data_term_matrix = matrix
    
    @property
    def data_frame_idf(self) -> pd.DataFrame():
        """
        - transform the idf matrix into panda data_frame 
        - created uppon request
        """
        if not self.__data_frame_idf:
            self.set_data_frame_idf()
        return self.__data_frame_idf

    @property
    def tf_idf_matrix(self) -> scipy.sparse.csr.csr_matrix:
        return self.__docs_matrix_tf_idf

    def set_data_frame_idf(self):
        """
        - create panda dataframe with idf weights
        - map of index to features
        """
        if not self.__applied:
            print("transformer: must call the 'apply' method first")
        elif not self.__transformer:
            print("transformer: transformer instance not found")
        else:
            self.__data_frame_idf = pd.DataFrame( {"feature": self.__feature_names,
                                                   "idf": self.__transformer.idf_} )
                                                   
            #sort descending
            self.__data_frame_idf.sort_values(by=['idf'],
                                              ascending=False,
                                              inplace = True)
        
    def set_transformer(self):
        self.__transformer = xCT.custom_transformer

    def fit(self):
        """ 
        - learns and computes the IDF vector
        - uses the sparse term matrix provided by the vectorization process
        - creates and caches the transformer instance
        """
        if not self.__transformer:
            self.set_transformer()

        if self.__data_term_matrix.size:
            self.__array_shape = self.__transformer.fit(X = self.__data_term_matrix)
            return True
        else:
            print('transformer: empty sparse matrix')
        return False
    
    def apply(self):
        """
        user callable method 
        """
        self.__applied = self.fit()


    def tf_idf_dataframe(self, args):
        """
        returns dataframe of tf-idf results
        create and using cached dataframe
       
        Note: must check if args are repeated
        """
        #if not self.__docs_frame_tf_idf_set:
        self.set_tf_idf_dataframe(args)
        return self.__docs_frame_tf_idf

    def set_tf_idf_dataframe(self, args=None):
        """
        Create a dataframe
        Sort grams by tf-idf score
        Transform scipy.sparse.csr.csr_matrix to pandas dataframe

        args:
          max, min, mean, median, ..
          sorted, ascending, descending

        no args:
          return complete sparse matrix as dataframe

        """
        if not args:
            self.__docs_frame_tf_idf = pd.DataFrame(self.__docs_matrix_tf_idf.toarray(),
                                                    columns = self.__feature_names)
            return True
            
        args  = [x.lower() for x in args]
        if "mean" in args:
            docs_array_tf_idf = np.asarray(self.__docs_matrix_tf_idf.mean(axis=0)).ravel().tolist()
        elif "min" in args:
            docs_array_tf_idf = np.asarray(self.__docs_matrix_tf_idf.min(axis=0)).ravel().tolist()
        elif "max" in args:
            docs_array_tf_idf = np.asarray(self.__docs_matrix_tf_idf.max(axis=0)).ravel().tolist()
        elif "sum" in args:
            docs_array_tf_idf = np.asarray(self.__docs_matrix_tf_idf.sum(axis=0)).ravel().tolist()
        else:
            docs_array_tf_idf = np.asarray(self.__docs_matrix_tf_idf.mean(axis=0)).ravel().tolist()
            
        self.__docs_frame_tf_idf = pd.DataFrame({'feature': self.__feature_names,
                                                 'tf-idf': docs_array_tf_idf})

        if "sorted" in args:                    
            ascending = True if "ascending" in args else False
            self.__docs_frame_tf_idf.sort_values(by='tf-idf',
                                                 ascending=ascending,
                                                 inplace = True)

        return True

    def compute_word_counts(self):
        """
        Get the word counts for provided documents in a sparse matrix form
        Use xVectorizer
        """
        self.__docs_matrix_word_count = self.__vectorizer.transform(self.__docs)

    def transform_word_counts(self):
        """
        Transform a count matrix to a tf or tf-idf representation using xTransformer
        Get back a sparse matrix of shape (n_samples, n_features)
        Matrix type: scipy.sparse.csr.csr_matrix
        """
        self.__docs_matrix_tf_idf = self.__transformer.transform(self.__docs_matrix_word_count)
        
    def compute_tf_idf(self, documents = None):
        """
        User callable method 
        - generate tf-idf scores for the provided document(s), i.e. compute the tf * idf 
        - when called as tf_idf(something), then 
        - when called as tf_idf(), then uses cached result
        """
        if not documents:
            print('compute_tf_idf: empty input')
            return False

        #check input type first - ensure it's a list
        if isinstance(documents, list):
            if all(isinstance(doc, str) for doc in documents):                
                self.__docs = documents
            else:
                print('compute_tf_idf: not a pure list of string objects')
                return False
        elif isinstance(documents, str):
            self.__docs = [documents]
            

        #actual computation
        self.compute_word_counts()
        self.transform_word_counts()

        #cache tf-idf DF only for new iqnuiries
        #and not for every call of this function
        self.__docs_frame_tf_idf_set = False

        return True

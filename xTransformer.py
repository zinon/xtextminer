import xCustomTransformer as xCT
import pandas as pd
import numpy as np
import scipy

class xTransformer():
    def __init__(self,
                 data_term_matrix = None,
                 feature_names = None):

        #NOTE: keep it central?
        #matrix of term/token counts
        self.__data_term_matrix = data_term_matrix

        #NOTE: keep it central?
        #names of features
        self.__feature_names = feature_names

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

    @property
    def data_term_matrix(self) -> scipy.sparse.csr.csr_matrix:
        return self.__data_term_matrix

    @data_term_matrix.setter
    def data_term_matrix(self, matrix:scipy.sparse.csr.csr_matrix = None):
        self.__data_term_matrix = matrix
    
    @property
    def data_frame_idf(self):
        """
        - transform the idf matrix into panda data_frame 
        - created uppon request
        """
        if not self.__data_frame_idf:
            self.set_data_frame_idf()
        return self.__data_frame_idf

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
            """
            self.__data_frame_idf = pd.DataFrame(self.__transformer.idf_,
                                                 index = self.__feature_names,
                                                 columns = ["idf_weights"])
            """
            self.__data_frame_idf = pd.DataFrame( {"feature": self.__feature_names,
                                                   "idf_weight": self.__transformer.idf_} )
                                                   
            #sort descending
            self.__data_frame_idf.sort_values(by=['idf_weight'],
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

    def compute_word_counts(self, TMP_CV):
        """
        get the word counts for provided documents in a sparse matrix form
        """
        self.__docs_matrix_word_count = TMP_CV.transform(self.__docs)

    def transform_word_counts(self):
        """
        Transform a count matrix to a tf or tf-idf representation
        Get back a sparse matrix of shape (n_samples, n_features)
        Create a dataframe
        Sort grams by tf-idf score
        """
        self.__docs_matrix_tf_idf = self.__transformer.transform(self.__docs_matrix_word_count)

        docs_array_tf_idf = np.asarray(self.__docs_matrix_tf_idf.mean(axis=0)).ravel().tolist()

        self.__docs_frame_tf_idf = pd.DataFrame({'gram': self.__feature_names,
                                                 'score': docs_array_tf_idf})

        self.__docs_frame_tf_idf.sort_values(by='score',
                                      ascending=False,
                                      inplace = True)

        print("TMP position")
        print(self.__docs_frame_tf_idf)
        
    def compute_df_idf(self, documents = None, TMP_CV=None):
        """
        User callable method 
        - generate tf-idf scores for the provided document(s), i.e. compute the tf * idf 
        - when called as df_idf(something), then 
        - when called as df_idf(), then uses cached result
        """
        if documents:
            self.__docs = documents
        
        self.compute_word_counts(TMP_CV)
        self.transform_word_counts()


from xVectorizer import xVectorizer
from xTransformer import xTransformer

class xTFIDF():
    def __init__(self, data = None):
        self.__data = data
        #
        self.__vectorizer = None

        self.__transformer = None

    def apply(self):
        #vectorize
        self.__vectorizer = xVectorizer(data_corpora=self.__data)
        self.__vectorizer.apply()
        #tranform
        self.__transformer = xTransformer(data_term_matrix = self.__vectorizer.data_term_matrix,
                                          feature_names = self.__vectorizer.feature_names)
        
        self.__transformer.apply()

    def tf(self):
        return self.__vectorizer.data_frame_tf
        
    def idf(self):
        return self.__transformer.data_frame_idf

    #NOTE: make it with input arg
    #either with doc label or custome text
    #otherwise consider the entire corpra
    def tfidf(self):
        self.__transformer.compute_df_idf(self.__data.texts, self.__vectorizer )

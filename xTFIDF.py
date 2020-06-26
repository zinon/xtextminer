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
        self.__transformer = xTransformer(vectorizer = self.__vectorizer)
        
        self.__transformer.apply()

    def tf(self):
        return self.__vectorizer.data_frame_tf
        
    def idf(self):
        return self.__transformer.data_frame_idf

    #NOTE: make it with input arg
    #either with doc label or custome text
    #otherwise consider the entire corpra
    def compute_tfidf(self, texts):
        self.__transformer.compute_df_idf(self.__data.texts)

    def tfidf_matrix(self):
        return self.__transformer.df_idf_matrix

    def tfidf_dataframe(self, *args):
        return self.__transformer.df_idf_dataframe(args)

                                       

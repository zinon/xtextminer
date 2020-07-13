from xVectorizer import xVectorizer
from xTransformer import xTransformer
from xSimilarity import xSimilarity

class xTFIDF():
    def __init__(self, corpora = None):
        self.__corpora = corpora
        #
        self.__vectorizer = None

        self.__transformer = None

        self.__similarities = None
        
    def apply(self):
        #vectorize
        self.__vectorizer = xVectorizer(data_corpora=self.__corpora)
        self.__vectorizer.apply()
        
        #tranform
        self.__transformer = xTransformer(vectorizer = self.__vectorizer)
        
        self.__transformer.apply()

        self.__similarity = xSimilarity()
        
    def tf(self):
        return self.__vectorizer.data_frame_tf
        
    def idf(self):
        return self.__transformer.data_frame_idf

    #NOTE: make it with input arg
    #either with doc label or custome text
    #otherwise consider the entire corpra
    def compute_tfidf(self, **kwargs):
        """
        Compute TF-IDF
        Input allowed:
        - text
        - xCorpus from xCorpora collectioned indexed by doc name
        - xCorpus from xCorpora collectioned indexed by doc index
        - Entire collection of xCorpora
        """

        if 'text' in kwargs:
            self.__transformer.compute_df_idf(kwargs['text'])
        elif 'name' in kwargs:
            self.__transformer.compute_df_idf( self.__corpora[kwargs['name']].text )
        elif 'index' in kwargs:
            self.__transformer.compute_df_idf( self.__corpora[kwargs['index']].text )
        else:       
            self.__transformer.compute_df_idf(self.__corpora.texts)

        #pipe sparse matrix to xSimilarities
        self.__similarity.matrix = self.__transformer.df_idf_matrix
        
    def tfidf_matrix(self):
        return self.__transformer.df_idf_matrix

    def tfidf_dataframe(self, *args):
        return self.__transformer.df_idf_dataframe(args)


    def tfidf_cosine_similarity_array(self):
        return self.__similarity.cosine_similarity_array

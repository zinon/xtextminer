from xVectorizer import xVectorizer
from xTransformer import xTransformer
from xSimilarity import xSimilarity
from xRepresentation import xRepresentation

class xTFIDF():
    def __init__(self, corpora = None):
        self.__corpora = corpora
        #
        self.__vectorizer = None

        self.__transformer = None

        self.__similarities = None

        self.__representation =  None
    def apply(self):
        #vectorize
        self.__vectorizer = xVectorizer(data_corpora=self.__corpora)
        self.__vectorizer.apply()
        
        #tranform
        self.__transformer = xTransformer(vectorizer = self.__vectorizer)
        
        self.__transformer.apply()

        #similarity
        self.__similarity = xSimilarity()

        #dimensionality reduction & representation
        self.__representation = xRepresentation(n_clusters=3,
                                                pca_components=3, #0.90
                                                tsne_components=3)
        
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
            self.__transformer.compute_tf_idf(kwargs['text'])
        elif 'name' in kwargs:
            self.__transformer.compute_tf_idf( self.__corpora[kwargs['name']].text )
        elif 'index' in kwargs:
            self.__transformer.compute_tf_idf( self.__corpora[kwargs['index']].text )
        else:       
            self.__transformer.compute_tf_idf(self.__corpora.texts)

        #pipe sparse matrix and doc names to xSimilarities
        self.__similarity.matrix_and_corpora_names( self.__transformer.tf_idf_matrix,
                                                    self.__corpora.names)

        #representation
        self.__representation.matrix = self.__transformer.tf_idf_matrix

        self.__representation.corpora_names = self.__corpora.names

        self.__representation.feature_names = self.__vectorizer.feature_names

        self.__representation.tfidf_dataframe = self.__transformer.tf_idf_dataframe(args=None)

        self.__representation.tfidf_angle_similarity_dataframe = \
            self.__similarity.angle_similarity_dataframe
        
    def tfidf_matrix(self):
        return self.__transformer.tf_idf_matrix

    def tfidf_dataframe(self, *args):
        return self.__transformer.tf_idf_dataframe(args=args)


    def tfidf_cosine_similarity_array(self):
        return self.__similarity.cosine_similarity_array

    def tfidf_angle_similarity_array(self):
        return self.__similarity.angle_similarity_array
    
    def tfidf_angle_similarity_dataframe(self):
        return self.__similarity.angle_similarity_dataframe

    def tfidf_angle_similarity(self, idx1 = None, idx2 = None):
        return self.__similarity.angle_similarity(idx1, idx2)

    def tfidf_most_similar(self, idx:str=None):
        return self.__similarity.most_similar(idx)
    
    def represent(self):
        self.__representation.fit()

    def clustered_terms_dataframe(self):
        return self.__representation.clusters.clustered_terms_dataframe

    def corpus_cluster(self, corpus_name:str):
        return self.__representation.clusters.corpus_cluster(corpus_name)

    def cluster_corpora(self, cluster:int):
        return self.__representation.clusters.cluster_corpora(cluster)
    

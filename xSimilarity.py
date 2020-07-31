from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

class xSimilarity:
    def __init__(self):
        self.__matrix = None
        self.__corpora_names = None
        #cached data
        self.__cosine_similarity_array = None
        self.__angle_similarity_array = None
        self.__angle_similarity_dataframe = None

    @property
    def matrix(self):
        return self.__matrix

    @matrix.setter
    def matrix(self, matrix):
        self.__clean_cache()
        self.__matrix = matrix

    def __clean_cache(self):
        self.__cosine_similarity_array = None
        self.__angle_similarity_array = None
        self.__angle_similarity_dataframe = None

    def matrix_and_corpora_names(self, matrix = None, names = None):
        if matrix.size:
            self.__matrix = matrix
        if names:
            self.__corpora_names = names
        
    @property
    def cosine_similarity_array(self):
        if self.__cosine_similarity_array is None:
            self.__set_cosine_similarity()
        return self.__cosine_similarity_array
    

    @property
    def angle_similarity_array(self):        
        if self.__angle_similarity_array is None:
            self.__set_angle_similarity_array()
        return self.__angle_similarity_array

    @property
    def angle_similarity_dataframe(self):
        if self.__angle_similarity_dataframe is None:
            self.__set_angle_similarity_dataframe()
        return self.__angle_similarity_dataframe

    def angle_similarity(self, idx1 = None, idx2 = None):
        if not self.__corpora_names:
            print('angle_similarity: empty corpora names list')
            return None

        #string index
        if isinstance(idx1, str):
            try:
                idx1 = self.__corpora_names.index(idx1)
            except ValueError:
                print(f"angle_similarity: invalid index '{idx1}'")
                return None

        #string index
        if isinstance(idx2, str):
            try:
                idx2 = self.__corpora_names.index(idx2)
            except ValueError:
                print(f"angle_similarity: invalid index '{idx2}'")
                return None

        return self.angle_similarity_array.item((idx1, idx2))

    def most_similar(self, idx:str):
        """
        select column based on document index name
        sort by tf-idf
        drop self-element
        return most similar document and score
        """
        series=self.angle_similarity_dataframe[idx].sort_values(ascending=False).drop(idx)
        return series.index[0], series[0]
        
    @staticmethod
    def __cosine_to_degrees(x):
        return np.nan_to_num(np.degrees(np.arccos(x)))

    def __set_cosine_similarity(self):
        self.__cosine_similarity_array = cosine_similarity(self.__matrix,
                                                           self.__matrix)
        
    def __set_angle_similarity_array(self):
        """
        express angle in radians
        transform radians to degrees
        """
        #from scipy import sparse
        #m = sparse.csr_matrix()
        self.__angle_similarity_array = self.__cosine_to_degrees(self.cosine_similarity_array)


    def __set_angle_similarity_dataframe(self):
        print(self.__corpora_names)
        self.__angle_similarity_dataframe = pd.DataFrame(data=self.angle_similarity_array,
                                                         index = self.__corpora_names,
                                                         columns = self.__corpora_names)



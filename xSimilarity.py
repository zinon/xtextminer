from sklearn.metrics.pairwise import cosine_similarity

class xSimilarity:
    def __init__(self):
        self.__matrix = None
        #cached data
        self.__cosine_similarity_array = None

    @property
    def matrix(self):
        return self.__matrix

    @matrix.setter
    def matrix(self, matrix):
        self.__clean_cache()
        self.__matrix = matrix

    def __clean_cache(self):
        self.__cosine_similarity_array = None
        
    @property
    def cosine_similarity_array(self):
        if not self.__cosine_similarity_array:
            self.__set_cosine_similarity()
        return self.__cosine_similarity_array
    
    def __set_cosine_similarity(self):
        self.__cosine_similarity_array = cosine_similarity(self.__matrix,
                                                           self.__matrix)

    

#angle_in_radians = math.acos(cos_sim)
#print math.degrees(angle_in_radians)

    

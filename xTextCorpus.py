from typing import List
from  xDataProvider import  xDataProvider

class xCorpus(object):
    def __init__(self, name:str = None, text:str = None):
        self.__name = name
        self.__text = text

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name:str = None):
        self.__name = name
        
    @property
    def text(self):
        return self.__text

    @text.setter
    def text(self, text:str = None):
        self.__text = text

class xCorpora(xDataProvider):
    def __init__(self, name:str = None):
        #
        super().__init__(name)
        #
        self.__corpora = list()
        self.__texts = list()
        self.__names = list()

    def add(self, corpus:xCorpus()):
        self.__corpora.append(corpus)

    @property
    def corpora(self) -> List[xCorpus]:
        return self.__corpora

    @property
    def texts(self) -> List[str]:
        """
        - get list of text
        - created upon request
        """
        if not self.__texts:
            self.__texts = [ corpus.text for corpus in self.__corpora]
        return self.__texts

    @property
    def names(self) -> List[str]:
        """
        - get list of names
        - created upon request
        """
        if not self.__names:
            self.__names = [ corpus.name for corpus in self.__corpora]
        return self.__names

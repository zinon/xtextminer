from typing import List
from  xDataProvider import  xDataProvider

class xCorpus:
    def __init__(self, name:str = None, text:str = None):
        self.__name = name
        self.__text = text

    def __repr__(self):
        return f'xCorpus({self.__name}, {self.__text})'

    def __str__(self):
        return self.description()

    def __eq__(self, other):
        return self.name == other.name and self.text == other.text
    
    def description(self):
        return f'{self.__name}: {self.__text}' \
            if self.__name and self.__text else "None"
    
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
        self.__new_item_added_cache_texts = False
        self.__new_item_added_cache_names = False

    def __len__(self):
        return len(self.__corpora)

    def __str__(self):
        return '\n'.join([x.description() for x in self.__corpora if x])
    
    def __iadd__(self, other):
        self.__corpora.extend(other.corpora)
        self.__texts.extend(other.texts)
        self.__names.extend(other.names)
        self.__notify(texts=True, names=True)
        return self

    def __getitem__(self, index):
        if isinstance(index, int):
            try:
                return self.__corpora[index]
            except IndexError:
                return None
        elif isinstance(index, str):
            for corpus in self.__corpora:
                if corpus.name == index:
                    return corpus
        elif isinstance(index, xCorpus):
            for corpus in self.__corpora:
                if corpus == index:
                    return corpus
                
        return None
        
    
    def __notify(self, **kwargs):
        """
        Notify actions for caching
        """
        if 'texts' in kwargs:
            self.__new_item_added_cache_texts = kwargs['texts']

        if 'names' in kwargs:
            self.__new_item_added_cache_names = kwargs['names']

    @property
    def corpora(self) -> List[xCorpus]:
        return self.__corpora

    @property
    def texts(self) -> List[str]:
        """
        - get list of text
        - created upon request
        """
        if self.__texts is None or self.__new_item_added_cache_texts:
            self.__texts = [ corpus.text for corpus in self.__corpora]
            self.__notify(texts=False)
        return self.__texts

    @property
    def names(self) -> List[str]:
        """
        - get list of names
        - created upon request
        """
        if self.__names is None or self.__new_item_added_cache_names:
            self.__names = [corpus.name for corpus in self.__corpora]
            self.__notify(names=False)
        return self.__names

    def add(self, corpus:xCorpus(), position:int = None):
        """
        User callable function to add new corpora
        """
        if corpus.name in self.names:
            raise ValueError("Cannot add corpus with existing name '{}'".format(corpus.name))
        self.__notify(texts=True, names=True)
        self.__corpora.append(corpus)

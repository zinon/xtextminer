import spacy
from typing import Dict


#class xStopwords():
# TBD    

class xSpacy:
    def __init__(self, library:str = 'small'):
        self.__library    = self.set_library(library)
        self.__nlp        = spacy.load(self.__library)
        self.__lemmatizer = spacy.lang.en.English()
        self.__tokenizer  = self.__nlp.Defaults.create_tokenizer(self.__nlp)
        self.__stopwords  = self.__nlp.Defaults.stop_words
    
    def set_library(self, lib:str) -> str:
        lib = lib.lower()
        if lib == 'small':
            return 'en_core_web_sm'
        elif lib == 'medium':
            return 'en_core_web_md'
        elif lib == 'large':
            return 'en_core_web_lg'
        else:
            print("xspacy: unknown vocabuly size '%s'. Returning default."%lib)
        return "en"

    def update_stopwords(self, words:Dict[str, bool]) -> None:
        #add new stopwords
        #words = { "SAP":True, "becomes":False}

        for key, value in words.items():
            nlp.vocab[key].is_stop = value
            
    def show_stopwords(self):
        print('Number of stop words: %d' % len(self.__stopwords))
        print('First ten stop words: %s' % list(self.__stopwords)[:10])
        print(self.__stopwords)
        
    @property
    def library(self):
        return self.__library

    @property
    def nlp(self):
        return self.__nlp

    @property
    def lemmatizer(self):
        return self.__lemmatizer

    @property
    def tokenizer(self):
        return self.__tokenizer

    @property
    def stopwords(self):
        return self.__stopwords

import spacy
# load spacy vocubulary
#spacy_lib = 'en'

class xSpacy():
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
            print("xspacy: unknown vocabuly size '%s'. Returning small edition."%lib)
        return ""
        
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

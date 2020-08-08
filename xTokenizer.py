from xSpacy import xSpacy
import re, string

xs = xSpacy()

class xTokenizer:
    def __init__(self):
        self.__digits = re.compile('\d')

    def __call__(self, text):
        """
        - tokenize a single string or a list of strings
        - allow tokens/words
        - called by CountVectorizer, also on stopwords
        """
        if isinstance((text), (str)):
            return self.process(text)
        elif isinstance((text), (list)):
            tokens = []
            for i in range(len(text)):
                tokens.append( self.process(text[i]) )
            return(tokens)
        else:
            print('tokenizer: cannot process a non-string-like parsed arg')
        return None

    def process(self, text:str):
        """
        select process method
        """
        return self.process0(text)

    def contains_digits(text:str):
        """
        identify if string contains digits
        """
        return bool(self.___digits.search(text))

    def allowed(self, token):
        """
        https://spacy.io/api/token#attributes
        """
        return not (token.is_stop or \
                    token.like_num or \
                    token.is_punct or \
                    token.is_space or \
                    token.is_digit or \
                    len(token)==1)

    def process0(self, text):
        """
        spacy doc --> lemmas and check for eligibility
        additional string cleaning 
        """
        doc = xs.nlp(text)
        lemmas = [ self.clean(token.lemma_) for token in doc if self.allowed(token) ]
        return(lemmas)

    @staticmethod
    def process1(text):
        """
        spacy doc --> lemmas 
        """
        doc = xs.nlp(text)
        lemmas = [token.lemma_ for token in doc]
        return lemmas
        
    def process2(self, text):
        """
        spacy tokenizer --> lemmas
        """
        tokens = xs.tokenizer(text)
        lemmas = [ token.lemma_ for token in tokens if self.allowed(token) ]
        return(lemmas)

    @staticmethod
    def process3(text:str):
        """
        spacy lemmatizer --> tokens
        """
        lemmas = xs.lemmatizer(text)
        tokens = [lemma.lemma_ for lemma in lemmas if not lemma.is_stop]
        return(tokens)

    @staticmethod
    def clean(text:str) -> str:
        """
        remove surrounding punctuation
        """
        return text.strip(string.punctuation)



from xSpacy import xSpacy

class xTokenizer(object):
    def __init__(self):
        pass

    def __call__(self, text):
        """
        tokenize a single string or a list of strings 
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
        return self.process2(text)
    
    def process1(self, text:str):
        """
        spacy lemmatizer --> tokens
        """
        lemmas = spacy_lemmatizer(text)
        tokens = [lemma.lemma_ for lemma in lemmas if not lemma.is_stop]
        return(tokens)

    
    def process2(self, text):
        """
        spacy tokenizer --> lemmas
        """

        tokens = spacy_tokenizer(text)
        lemmas = [ token.lemma_ for token in tokens if not token.is_stop]
        return(lemmas)
    

from html import unescape
import re
import unicodedata
import string

class xPreprocessor:
    def __init__(self):
        pass

    def __call__(self, text):
        """
        clean a single string or a list of strings 
        """
        if isinstance((text), (str)):
            return self.clean(text)
        elif isinstance((text), (list)):
            proc_list = []
            for i in range(len(text)):
                proc_list.append( self.clean(text[i]) )
            return(proc_list)
        else:
            print('preprocessor: cannot process a non-string-like parsed arg')
        return None

    def clean(self, text = None) -> str:
        """
        Cleaning text, not tokens
        - function order is important
        - the following spoil lemmatization during tokenization process:
           - punctuation removal
           - special character removal
        - can remove numeric in tokenization
        """
        return self.normalize(
            self.clean_double_whitespaces(
                self.clean_edge_punctuation(
                    self.clean_numeric(
                        self.clean_web_tags(
                            self.clean_html_tags(
                                self.clean_non_ascii(
                                    text
                                )
                            )
                        )
                    )
                )
            )
        )

    @staticmethod
    def clean_edge_punctuation(text:str):
        """
        clean string edges 
        """
        return text.strip(string.punctuation)
    
    @staticmethod
    def clean_punctuation(text:str):
        """
        clean string edges 
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def clean_special_characters(text):
        """
        #@ etc
        """
        return re.sub(r'[^a-zA-z\s]', '', text)
    
    @staticmethod
    def clean_non_ascii(text:str):
        """
        replace non-ascii symbols with space
        """
        return re.sub(r'[^\x00-\x7F]+',' ', text)
        
    @staticmethod
    def clean_accented(text):
        """
        accented characters/letters are converted and standardized into ASCII characters
        eg. convert é to e
        """
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        
    @staticmethod
    def normalize(text):
        """ lowercase """
        return text.lower()

    @staticmethod
    def clean_web_tags(text):
        """remove web tags"""
        return re.sub('<[^>]*>', ' ', text)

    @staticmethod
    def clean_html_tags(text):
        """remove html tags """
        return unescape(text)

    @staticmethod
    def clean_punctuation(text):
        """remove punctuation"""
        #return re.sub(r'[^\w\s]','', text)
        return re.sub('[\W]+', ' ', text)
    
    @staticmethod
    def clean_double_whitespaces(text):
        """
         re.sub(r"\s+"," ", text, flags = re.I)
        """
        return re.sub(' +', ' ', text)

    @staticmethod
    def clean_numeric(text):
        return re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)

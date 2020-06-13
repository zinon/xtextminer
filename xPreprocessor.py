from html import unescape
import re

class xPreprocessor(object):
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

    def clean(self, text):
        """
        function order is important
        """
        return self.normalize(
            self.clean_double_whitespaces(
                self.clean_punctuation(
                    self.clean_web_tags(
                        self.clean_html_tags(
                            text
                        )
                    )
                )
            )
        )

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
        return re.sub(' +', ' ', text)

from sklearn.feature_extraction.text import CountVectorizer

from xPreprocessor import xPreprocessor
from xTokenizer import xTokenizer

### must be put in a base class


    

# tokenize the doc and lemmatize its tokens
def tokenizer1(doc):
    #tokens
    tokens = tokenizer(doc)

    #lemmas
    lemma_list = []
    for token in tokens:
        #avoid stopwords
        if token.is_stop is False:
            lemma_list.append(token.lemma_)
     
    return(lemma_list)

def tokenizer2(doc):

    tokens = lemmatizer(doc)
    return([token.lemma_ for token in tokens])


preprocessor = xPreprocessor()
tokenizer = xTokenizer()


custom_vectorizer = CountVectorizer(preprocessor = preprocessor,
                                    tokenizer = tokenizer)

#custom_vectorizer = CountVectorizer()

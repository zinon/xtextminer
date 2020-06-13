from sklearn.feature_extraction.text import CountVectorizer
import spacy
from xPreprocessor import xPreprocessor

### must be put in a base class

# load spacy vocubulary
#spacy_lib = 'en'
spacy_lib = 'en_core_web_sm'
nlp = spacy.load(spacy_lib)
lemmatizer = spacy.lang.en.English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)
spacy_stopwords = nlp.Defaults.stop_words

    

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


custom_vectorizer = CountVectorizer(preprocessor = preprocessor,
                                    tokenizer = tokenizer2)

#custom_vectorizer = CountVectorizer()

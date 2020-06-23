"""
Under the hood, Sklearn’s vectorizers call a series of functions to convert a set of
documents into a document-term matrix. Out of which, three methods stand out:
- build_preprocessor: Returns a callable utilized to preprocess the input text before tokenization.
- build_tokenizer: Creates a function capable of splitting a document’s corpus into tokens.
- build_analyzer: Builds an analyzer function which applies preprocessing, tokenization, remove stop words and creates n-grams.

"""
from sklearn.feature_extraction.text import CountVectorizer

from xPreprocessor import xPreprocessor
from xTokenizer import xTokenizer
from xSpacy import xSpacy
#text preprocessor
#preprocessor = 

#tokenizer
#tokenizer = 


# ignore words that were too rare with MIN_DF
min_df = 0.10

#ignore words that are too common with MAX_DF
max_df = 0.90

#Limiting Vocabulary Size - keep most frequent n-grams and drop rest
max_features = 1000

#Ignore Counts & get Binary Values
#If a token is present in a document, it is 1,
#if absent it is 0 regardless of its frequency of occurrence. 
binary = False

#Word level / N-grams
# unigrams 1,1
# bigrams 2,2
# unigrams and bigrams 1,2
ngram_range = (1, 1)

#import spacy
#nlp = spacy.load("en")


#stopwords
#stopwords = nlp.Defaults.stop_words
#stopwords.add("brown")
#nlp.vocab["brown"].is_stop = True

#nlp.Defaults.stop_words.add("brown")
#nlp.vocab["brown"].is_stop = True

#stopwords = ['additional', 'brown']

xs = xSpacy()
stopwords = xs.stopwords


custom_vectorizer = CountVectorizer(preprocessor = xPreprocessor(),
                                    tokenizer = xTokenizer(),
                                    #analyzer = analyzer,
                                    #min_df = min_df,
                                    #max_df = max_df,
                                    stop_words = stopwords,
                                    #max_features = max_features,
                                    binary = binary,
                                    lowercase = True,
                                    ngram_range=ngram_range 
)


#token_pattern=r"(?u)\b\w\w+\b"
#custom_vectorizer = CountVectorizer(stop_words=N)

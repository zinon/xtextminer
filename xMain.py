from xTextCorpus import xCorpora, xCorpus
from xVectorizer import xVectorizer
from xTransformer import xTransformer


xc = xCorpora("test")
xc.add( xCorpus(name='doc1', text='The   quick brown fox&#x0002E;') )
xc.add( xCorpus(name='doc2', text="jumped over the lazy dog's back&#x00021; It's brown") )
xc.add( xCorpus(name='doc3', text='<div class="a div class"> The brown  class</div>') )
xc.add( xCorpus(name='doc4', text="I'm jumping while he's lying. We've 400 cases.") )


xv = xVectorizer(data=xc)


xv.apply()
print(xv)

#print(xv.term_frequency_matrix_stats())
#print(xv.stopwords())
#print(xv.vocabulary())
print(xv.frequencies)

xt = xTransformer(sparse_term_matrix = xv.sparse_term_matrix,
                  feature_names = xv.feature_names)

xt.apply()
print(xt.data_frame_idf)

xt.compute_df_idf( xc.texts, xv )

from xDataLoader import xDataLoader

xdl = xDataLoader(catalogues=['catalogues/s4hana.json'])
#xdl = xDataLoader(catalogues=['catalogues/test.json'])
xdl.load(summary=True)
xc = xdl.corpora

#from test_corpora import *
#xc=xc1

##################################################################################3
from xTFIDF import xTFIDF
xdf = xTFIDF(corpora=xc)
xdf.apply()


print(xdf.tf())
print(xdf.idf())

#NOTE here add input, otherwise will consider entire doc
# entire doc
# specific doc (key)
# custom text

xdf.compute_tfidf()
#xdf.compute_tfidf(name='doc4')

mat = xdf.tfidf_matrix()
print(xdf.tfidf_dataframe("mean", "sorted", "descending"))
print(xdf.tfidf_dataframe())

#print(xdf.tfidf_cosine_similarity_array())
#print(xdf.tfidf_angle_similarity_array())
#print(xdf.tfidf_angle_similarity_dataframe())
#print(xdf.tfidf_angle_similarity('ref1', 'doc2'))
#print(xdf.tfidf_most_similar('ref1'))

xdf.represent()

print(xdf.clustered_terms_dataframe())
print(xdf.corpus_cluster('doc1'))
print(xdf.cluster_corpora(1))


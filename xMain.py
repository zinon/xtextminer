from xTextCorpus import xCorpora, xCorpus

xc = xCorpora("testme")
xc.add( xCorpus(name='doc1', text='The   quickest brown fox&#x0002E;') )
xc.add( xCorpus(name='doc2', text="jumped over the lazy dog's back&#x00021; It's brown and shouldn't") )
xc.add( xCorpus(name='doc3', text='<div class="a div class"> The brown class </div>') )
xc.add( xCorpus(name='doc4', text="I'm jumping while he's lying. We've got 400 brown cases.") )
#xc.add( xCorpus(name='doc4', text="Cannot add a corpus with a name existing already!") )
print("xc length", len(xc))



xc2 = xCorpora("tryme")
xc2.add( xCorpus(name='doc5', text='The yellow wolf lives in the forest') )
xc2.add( xCorpus(name='doc6', text='Yellow wolf, brown fox, gray dog are all animals') )
xc2.add( xCorpus(name='doc7', text='Lazy students sitting in the class') )
print("xc2", len(xc2))

xc += xc2
print("xc", len(xc))

print(xc)
print(25*'=')
print(xc[1])
print(xc['doc3'])

x4 = xCorpus(name='doc4', text="I'm jumping while he's lying. We've got 400 brown cases.")
print(xc[x4])

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

print(xdf.tfidf_cosine_similarity_array())
print(xdf.tfidf_angle_similarity_array())
print(xdf.tfidf_angle_similarity_dataframe())
print(xdf.tfidf_angle_similarity('doc1', 'doc2'))
print(xdf.tfidf_most_similar('doc1'))

xdf.represent()

print(xdf.clustered_terms_dataframe())
print(xdf.corpus_cluster('doc5'))
print(xdf.cluster_corpora(1))


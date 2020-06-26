from xTextCorpus import xCorpora, xCorpus

xc = xCorpora("test")
xc.add( xCorpus(name='doc1', text='The   quickest brown fox&#x0002E;') )
xc.add( xCorpus(name='doc2', text="jumped over the lazy dog's back&#x00021; It's brown and shouldn't") )
xc.add( xCorpus(name='doc3', text='<div class="a div class"> The brown class </div>') )
xc.add( xCorpus(name='doc4', text="I'm jumping while he's lying. We've got 400 brown cases.") )
print("xc", len(xc))

xc2 = xCorpora("try")
xc2.add( xCorpus(name='doc5', text='The yellow wolf') )
print("xc2", len(xc2))
xc += xc2
print("xc", len(xc))

print(xc)

from xTFIDF import xTFIDF
xdf = xTFIDF(data=xc)
xdf.apply()
print(xdf.tf())
print(xdf.idf())

#NOTE here add input, otherwise will consider entire doc
xdf.compute_tfidf("")
print(xdf.tfidf_matrix)
print(xdf.tfidf_dataframe("mean", "sorted", "descending"))
print(xdf.tfidf_dataframe())

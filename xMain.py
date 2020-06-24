from xTextCorpus import xCorpora, xCorpus

xc = xCorpora("test")
xc.add( xCorpus(name='doc1', text='The   quickest brown fox&#x0002E;') )
xc.add( xCorpus(name='doc2', text="jumped over the lazy dog's back&#x00021; It's brown and shouldn't") )
xc.add( xCorpus(name='doc3', text='<div class="a div class"> The brown class </div>') )
xc.add( xCorpus(name='doc4', text="I'm jumping while he's lying. We've 400 cases.") )


from xTFIDF import xTFIDF
xdf = xTFIDF(data=xc)
xdf.apply()
print(xdf.tf())
print(xdf.idf())
#NOTE here add input, otherwise will consider entire doc
xdf.tfidf()


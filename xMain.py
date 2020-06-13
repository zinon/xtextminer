from xVectorizer import xVectorizer
from xTextCorpus import xCorpora, xCorpus



xc = xCorpora("test")
xc.add( xCorpus(name='doc1', text='The quick brown fox&#x0002E;') )
xc.add( xCorpus(name='doc2', text='jumped over the lazy dog&#x00021;') )
xc.add( xCorpus(name='doc3', text='<div class="This is a div class">Test</div>') ) 


xv = xVectorizer(data=xc)
xv.apply()
print(xv)

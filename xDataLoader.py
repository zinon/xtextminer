import json, os
from typing import List
from xTextCorpus import xCorpora, xCorpus

class xDataLoader:
    def __init__(self,
                 catalogues:List[str] = None,
                 task:str = ""):
        self.__catalogues = catalogues
        self.__task = task
        #
        self.__data_json = list()
        self.__corpora = xCorpora(self.__task)

    @property
    def corpora(self):
        return self.__corpora
    
    @staticmethod
    def valid_file(fname:str):
        return os.path.isfile(fname)

    def check(self):
        for catalogue in self.__catalogues:
            if not self.valid_file(catalogue):
                print(f'{catalogue} not found!')
                return False

        return True
                
    def read_json(self):
        for catalogue in self.__catalogues:
            with open(catalogue) as json_file:
                try:
                    self.__data_json.append(json.load(json_file))
                    print(f'{catalogue} registered!')
                except:
                    print(f'{catalogue} cannot be loaded!')
        return True
    
    def read_data(self):
        
        for data in self.__data_json:
            path = data['path']
            for file_name, metadata in data['documents'].items():
                
                if "enable" in metadata and not metadata["enable"]:
                    print(f"{doc_file_name} skipped")
                    continue                    

                if "name" in metadata and metadata["name"]:
                    name = metadata["name"]
                else:
                    print(f"name for {doc_file_name} not found")
                    return False

                doc_file_name = os.path.join(path, file_name)
                if not self.valid_file(doc_file_name):
                    print(f'name not found for file {doc_file_name} in {data}')
                    return False
                
                with open(doc_file_name) as doc:
                    front = 0 if "reference" in metadata and metadata["reference"] else None
                    self.__corpora.add(corpus = xCorpus(name=name, text=doc.read()),
                           position = None)

        return True

    def summary(self):
        print("Corpora size", len(self.__corpora))
        #print(self.__corpora)
    
    def load(self, summary = False):
        if not self.check():
            return False

        if not self.read_json():
            return False
        
        if not self.read_data():
            return False

        if summary:
            self.summary()

        return True

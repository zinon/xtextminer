from abc import ABC, abstractmethod, abstractproperty

class xDataProvider(ABC):
    """
    - interface to data lakes
    """
    def __init__(self, name:str):
        self._name = name

    #def __repr__(self):
    #    return f'{self.__class__.__name__}'
    
    @abstractproperty
    def texts(self):
        pass

    @abstractproperty
    def names(self):
        pass

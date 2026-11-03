from abc import abstractmethod, ABC


class Labeling(ABC):
    @abstractmethod
    def label(self):
        pass
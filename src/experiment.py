from abc import ABC
from json import Config

class Experiment(ABC):
    """Abstract Experiment class that is inherited to all experiments"""
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)
        

    @abstractmethod
    def load_data(self):
        """Define the dataset."""
        pass
    
    @abstractmethod
    def _preprocess_data(self):
        """Define the transformations and apply MTCNN if desired."""
        pass
    
    @abstractmethod
    def _load_image_train(self):
        """Define and load the train dataset."""
        pass

    @abstractmethod
    def _load_image_test(self):
        """Define and load the test dataset."""
        pass

    @abstractmethod
    def build(self):
        """Create model."""
        pass

    @abstractmethod
    def train(self):
        """Determine training routine, select which layers should be trained, and fit the model."""
        pass

    @abstractmethod
    def evaluate(self):
        """Predict results for test set and measure accuracy."""
        pass
from typing import List, Any
import json
from abc import  abstractmethod
from cog import BasePredictor, Input, Path
from pydantic import BaseModel

class VanaPredictorInput(BaseModel):
    """
    Base class for all predictor inputs
    """

class VanaPredictorOutput(BaseModel):
    """
    Base class for all predictor outputs
    """

class VanaPredictor():

    def setup(self):
        """
        Setup the predictor
        """

    @abstractmethod
    def run(self, input: VanaPredictorInput) -> Any:
        """
        Run a single prediction on the model
        """

    @abstractmethod
    def predict(self, **kwargs: Any) -> Any:
        """
        Run a single prediction on the model
        """

    def predict_api(self, input: Input(str)) -> Any:
        return self.run(
            self.input_class(**json.loads(input)))
    
    @abstractmethod
    def predict_ui(self, **kwargs: Any) -> Any:
        """
        Run a single prediction on the model
        """


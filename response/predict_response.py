from pydantic import BaseModel

class PredictResponse(BaseModel):
    turbo_frequency: float
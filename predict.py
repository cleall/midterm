import pickle
import uvicorn
import os

from fastapi import FastAPI
from response.cpu import Cpu
from response.predict_response import PredictResponse

MLR_PL = "cpu_tf_rfpl_v1.bin"

with open(MLR_PL, "rb") as pipe_in:
    pipeline = pickle.load(pipe_in)

def predict_single(observation):
    score = pipeline.predict(observation)
    return float(score)

predict_cpu_tf_rf = FastAPI(title="predict_cpu_tf_rf")

@predict_cpu_tf_rf.post("/predict_cpu_tf_rf")
def predict(observation: Cpu) -> PredictResponse:
    score = predict_single(observation.model_dump())
    
    return PredictResponse(
        turbo_frequency = score
    )

def main():
    PORT = int(os.environ.get("PORT", 4444))
    uvicorn.run("predict:predict_cpu_tf_rf", host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()
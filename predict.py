import pickle
import uvicorn
import os
import httpx

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from response.cpu import Cpu
from response.predict_response import PredictResponse

MLR_PL = "cpu_tf_rfpl_v1.bin"
#URL = "http://0.0.0.0:4444/predict_cpu_tf"
URL="https://predict-cpu-turbo-frequency.onrender.com/predict_cpu_tf"
FORM = "index.html"

predict_cpu_tf = FastAPI(title="predict_cpu_tf")
predict_cpu_tf.mount("/static", StaticFiles(directory="static"), name="static")
web_files = Jinja2Templates(directory="web")

with open(MLR_PL, "rb") as pipe_in:
    pipeline = pickle.load(pipe_in)

def predict_single(observation):
    score = pipeline.predict(observation)
    return float(score)

@predict_cpu_tf.get("/", response_class=HTMLResponse)
def root(request: Request):
    return web_files.TemplateResponse(FORM, {"request": request}, context={})

@predict_cpu_tf.post("/submit", response_class=HTMLResponse)
async def submit_form(request: Request):
    #form data as dictionary
    form_data = await request.form()
    observation = form_data._dict
    #send data to endpoint
    async with httpx.AsyncClient() as client:
        response = await client.post(URL, json=observation)

    #update form
    return web_files.TemplateResponse(FORM, {
        "request": request,
        "api_response": response.json()
    })

@predict_cpu_tf.post("/predict_cpu_tf")
def predict(observation: Cpu) -> PredictResponse:
    score = predict_single(observation.model_dump())
    
    return PredictResponse(
        turbo_frequency = score
    )

def main():
    PORT = int(os.environ.get("PORT", 4444))
    uvicorn.run("predict:predict_cpu_tf", host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()
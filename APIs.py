#!pip install fastapi
#!pip install "uvicorn[standard]"

from fastapi import FastAPI

app = FastAPI()
@app.get("/")
async def root():
  return check_json('/content/train_input_cv.json')

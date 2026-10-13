from fastapi import FastAPI

app = FastAPI(title="YOLO11")

@app.post("/predict")
def predict():
    pass
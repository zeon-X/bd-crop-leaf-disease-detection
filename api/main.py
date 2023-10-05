from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

import cv2

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "../saved_models/"


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    model_name: str = Form(...),
    file: UploadFile = File(...)
):
    model_path = MODEL_DIR + model_name + "_v1"

    try:
        MODEL = tf.keras.models.load_model(model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    image = read_file_as_image(await file.read())

    # Resize the image to match the expected input shape
    image = cv2.resize(image, (256, 256))

    img_batch = np.expand_dims(image, 0)

    classes = {
        "Rice": ["Rice___Brown_Spot", "Rice___Healthy", "Rice___Leaf_Blast", "Rice___Neck_Blast"],
        "Corn": ["Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy", "Corn___Northern_Leaf_Blight"],
        "Wheat": ["Wheat___Brown_Rust", "Wheat___Healthy", "Wheat___Yellow_Rust"],
        "Potato": ["Potato___Early_Blight", "Potato___Healthy", "Potato___Late_Blight"],
        "Sugarcane": ["Healthy", "RedRot", "RedRust"]
    }

    if model_name not in classes:
        return {"error": "Invalid model name"}

    CLASS_NAMES = classes[model_name]

    try:
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return {
            'class': predicted_class,
            'confidence': confidence
        }
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

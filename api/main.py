# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf

# import cv2

# app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# MODEL_DIR = "./saved_models/"


# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"


# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image


# @app.post("/predict")
# async def predict(
#     model_name: str = Form(...),
#     file: UploadFile = File(...)
# ):
#     model_path = MODEL_DIR + model_name + "_v1"

#     try:
#         # MODEL = tf.keras.models.load_model(model_path)
#         MODEL = tf.saved_model.load(model_path)
#     except Exception as e:
#         print("Model Loading Error:", e)  # Debug statement
#         return {"error": f"Failed to load model: {e}"}

#     image = read_file_as_image(await file.read())

#     # Resize the image to match the expected input shape
#     image = cv2.resize(image, (256, 256))

#     img_batch = np.expand_dims(image, 0)

#     classes = {
#         "Rice": ["Rice___Brown_Spot", "Rice___Healthy", "Rice___Leaf_Blast", "Rice___Neck_Blast"],
#         "Corn": ["Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy", "Corn___Northern_Leaf_Blight"],
#         "Wheat": ["Wheat___Brown_Rust", "Wheat___Healthy", "Wheat___Yellow_Rust"],
#         "Potato": ["Potato___Early_Blight", "Potato___Healthy", "Potato___Late_Blight"],
#         "Sugarcane": ["Healthy", "RedRot", "RedRust"]
#     }

#     if model_name not in classes:
#         return {"error": "Invalid model name"}

#     CLASS_NAMES = classes[model_name]

#     try:
#         predictions = MODEL.predict(img_batch)
#         predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#         confidence = float(np.max(predictions[0]))
#         return {
#             'class': predicted_class,
#             'confidence': confidence
#         }
#     except Exception as e:
#         print(f"failed to predict: {e}")
#         return {"error": f"Prediction failed: {e}"}

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2

app = FastAPI()

# CORS setup
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

# Directory where SavedModels are stored
MODEL_DIR = "./saved_models/"

# Class names corresponding to different model outputs
CLASS_NAMES = {
    "Rice": ["Brown Spot", "Healthy", "Leaf Blast", "Neck Blast"],
    "Corn": ["Common Rust", "Gray Leaf Spot", "Healthy", "Northern Leaf Blight"],
    "Wheat": ["Brown Rust", "Healthy", "Yellow Rust"],
    "Potato": ["Early Blight", "Healthy", "Late Blight"],
    "Sugarcane": ["Healthy", "Red Rot", "Red Rust"]
}


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
        # Load the SavedModel
        model = tf.saved_model.load(model_path)

        # Preprocess the uploaded image
        image = read_file_as_image(await file.read())
        image = cv2.resize(image, (256, 256))
        img_batch = np.expand_dims(image, 0)

        # Perform inference using the model
        output = model(img_batch.astype(np.float32))

        # Convert the model output to probabilities
        probabilities = tf.nn.softmax(output)

        # Determine the predicted class and confidence
        predicted_class_index = np.argmax(probabilities[0])
        confidence = float(probabilities[0][predicted_class_index])

        # Get the predicted class name
        model_classes = CLASS_NAMES.get(model_name)
        if model_classes is None:
            return {"error": "Invalid model name"}

        predicted_class = model_classes[predicted_class_index]

        # Prepare the response with all class confidences
        class_confidences = [
            {"class": CLASS_NAMES.get(model_name)[
                i], "confidence": float(probabilities[0][i])}
            for i in range(len(probabilities[0]))
        ]

        return {
            'class': predicted_class,
            'confidence': confidence,
            'class_confidences': class_confidences
        }
    except Exception as e:
        print(f"Prediction failed: {e}")
        return {"error": f"Prediction failed: {e}"}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

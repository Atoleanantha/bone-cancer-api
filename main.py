
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow import keras
import numpy as np
import os
import cv2
from PIL import Image
import shutil
import base64
from openai import OpenAI

from fastapi.middleware.cors import CORSMiddleware


client = OpenAI(api_key="sk-your-api-key")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"  # Temporary folder for image uploads
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model globally (only once)
MODEL_PATH = "static/new_5_6_bone_cancer_inception_best_weights.keras"
model = keras.models.load_model(MODEL_PATH)
class_labels = model.class_names if hasattr(model, 'class_names') else ['benign','chondrosarcoma', 'ewingsarcoma', 'osteosarcoma', 'other']

# # Preprocess the image
# def preprocess_image(image_path):
#     image = Image.open(image_path).resize((224, 224))
#     image_array = keras.preprocessing.image.img_to_array(image)
#     image_array = np.expand_dims(image_array, axis=0)
#     image_array = keras.applications.vgg16.preprocess_input(image_array)
#     return image_array

# def preprocess_image(image_path):
#     """Apply preprocessing and improved segmentation on an image."""
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.resize(image, (299, 299))  # Resize to match InceptionV3 input
#     image = cv2.GaussianBlur(image, (5, 5), 0)  # Apply Gaussian Blur to remove noise
#     image = cv2.equalizeHist(image)  # Improve contrast
    
#     # Apply adaptive histogram equalization (CLAHE)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     image = clahe.apply(image)
    
#     # Apply adaptive thresholding for better segmentation
#     thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                    cv2.THRESH_BINARY, 11, 2)
    
#     # Apply Canny edge detection to detect tumor boundaries
#     edges = cv2.Canny(image, 50, 150)
    
#     # Find contours and draw the largest one (assumed tumor area)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     mask = np.zeros_like(image)
#     cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    
#     # Apply the mask to segment the tumor area
#     segmented_image = cv2.bitwise_and(image, image, mask=mask)
    
#     # Normalize pixel values
#     segmented_image = segmented_image / 255.0
    
#     image_array = keras.preprocessing.image.img_to_array(segmented_image)
#     image_array = np.expand_dims(image_array, axis=0)
#     image_array = preprocess_input(image_array)
#     return image_array
#     return segmented_image

def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Read in color (default)
    image = cv2.resize(image, (299, 299))
    image = image / 255.0  # Normalize to [0, 1]
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Apply Gaussian Blur to remove noise
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array

# Prediction function
def predict_image(file_path):
    processed_image = preprocess_image(file_path)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_labels[predicted_class_index]
    confidence = np.max(predictions) * 100  # Convert to percentage
    return predicted_class_name, confidence

async def is_valid_xray(base64_image: str) -> bool:
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Is this a valid X-ray image? Just answer 'Yes' or 'No'."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer
    except Exception:
        return False

@app.get("/")
def api_root():
    return {"message": "Bone cancer prediction api running"}

@app.post("/detect-cancer")
def detect():
    return {"message":"detecting..."}

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, image.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        prediction_class, confidence = predict_image(file_path)
        os.remove(file_path)  # Clean up the uploaded file
        
        return {"prediction": prediction_class, "confidence": f"{confidence:.2f}%"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


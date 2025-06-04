import shutil
import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import tensorflow as tf

app = FastAPI()

# HTML template directory
templates = Jinja2Templates(directory="templates")

# Directory for uploaded images
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load TB and Pneumonia models from models/ directory
model_paths = {
    "2": "models/tb_cnn_model.h5",
    "3": "models/chest_xray_cnn_model.h5"
}

models = {}
for key, path in model_paths.items():
    models[key] = tf.keras.models.load_model(path)
    print(f"Loaded model for choice {key}: {path} with input shape {models[key].input_shape}")

# Save uploaded file
def save_uploaded_file(file, destination):
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

# Image preprocessing
def preprocess_image(image_path, target_size=(150, 150)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # shape: (1, 150, 150, 3)
    return img

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Upload and classify
@app.post("/upload/", response_class=HTMLResponse)
async def upload_file(request: Request, choice: str = Form(...), data: UploadFile = Form(...)):
    file_location = os.path.join(UPLOAD_DIR, "uploaded_image.jpg")
    save_uploaded_file(data, file_location)

    img = preprocess_image(file_location)

    if choice not in models:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid choice selected."})

    model = models[choice]
    prediction_prob = model.predict(img)[0][0]
    print(f"Prediction score: {prediction_prob}")

    if choice == "2":
        result = "TB Detected" if prediction_prob >= 0.5 else "Normal"
    elif choice == "3":
        result = "Pneumonia Detected" if prediction_prob >= 0.5 else "Normal"
    else:
        result = "Invalid"

    return templates.TemplateResponse("result.html", {"request": request, "result": result, "choice": choice})

# Run app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

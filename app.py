from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

# Force TensorFlow to use CPU only to avoid GPU errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = FastAPI(title="Sign Language Detection", description="Real-time sign language detection using CNN")
templates = Jinja2Templates(directory="templates")

# Load the trained model
model = tf.keras.models.load_model('sign_mnist.h5')

# Sign language alphabet mapping (A-Z, no J or Z in MNIST dataset)
SIGN_LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

def preprocess_image(image_data):
    """Preprocess image for model prediction"""
    # Decode base64 image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to grayscale
    image = image.convert('L')

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to numpy array
    image_array = np.array(image)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Reshape for model input
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_sign(request: dict):
    """Predict sign language letter from image"""
    try:
        image_data = request.get('image')
        if not image_data:
            return JSONResponse({"success": False, "error": "No image data provided"})

        # Preprocess the image
        processed_image = preprocess_image(image_data)

        # Make prediction
        prediction = model.predict(processed_image, verbose=0)

        # Get the predicted class and confidence
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))

        # Map to letter
        predicted_letter = SIGN_LABELS.get(predicted_class, 'Unknown')

        return JSONResponse({
            "success": True,
            "letter": predicted_letter,
            "confidence": confidence,
            "class_id": int(predicted_class)
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
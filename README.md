# Sign Language Detector - Real-time ASL Recognition

A FastAPI-based American Sign Language (ASL) detector that recognizes hand gestures in real-time using computer vision and deep learning. The system converts ASL alphabetic letters (A-Z, excluding J and Z which require motion) into text using a Convolutional Neural Network trained on the Sign Language MNIST dataset.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam for real-time detection
- TensorFlow 2.x compatible system

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Sign_Language_Detector-PyTorch
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### FastAPI Web Application (Recommended)
```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```
Access the application at http://localhost:8080/

#### Alternative: Direct Python execution
```bash
python app.py
```
Runs on http://localhost:8000/

#### Standalone OpenCV Application
```bash
python main.py
```
Opens OpenCV window with real-time detection. Press 'q' to quit.

## Screenshots
### H
![H](https://user-images.githubusercontent.com/34855465/76798612-eda6c700-67f5-11ea-974e-514a82c8c5c5.png)

### A
![A](https://user-images.githubusercontent.com/34855465/76798664-044d1e00-67f6-11ea-9b41-0a4ca9f625e1.png)

## Demo Video
https://youtu.be/70nmZY5ASvw

## Inspiration
There are only 250 certified sign language interpreters in India, translating for a deaf population of between 1.8 million and 7 million.

![percentage](https://user-images.githubusercontent.com/34855465/76789152-42404700-67e2-11ea-8e96-718ba4ae0a36.png)

We need to devise a solution that allows inclusion of the deaf and mute people in normal conversations. Our application allows any user to point the camera towards a mute person (with consent, ofcourse) and effectively understand what he/she's trying to say.

### American Sign Language (ASL)
American Sign Language (ASL) is a visual language. With signing, the brain processes linguistic information through the eyes. The shape, placement, and movement of the hands, as well as facial expressions and body movements, all play important parts in conveying information. 
![ASL](https://user-images.githubusercontent.com/34855465/76790591-28ecca00-67e5-11ea-990d-b6540acb9a1b.png)


## 📊 Technical Architecture

### Model Details
- **Framework:** TensorFlow/Keras (migrated from PyTorch)
- **Architecture:** Convolutional Neural Network (CNN)
- **Input:** 28×28 grayscale images
- **Output:** 24 classes (A-Y, excluding J and Z)
- **Model File:** `sign_mnist.h5`

### Image Processing Pipeline
1. **Capture:** Real-time video from webcam
2. **Preprocessing:**
   - Resize to 28×28 pixels
   - Convert to grayscale
   - Normalize pixel values (0-1 range)
3. **Inference:** CNN model prediction
4. **Post-processing:** Confidence thresholding and label mapping

### API Endpoints
- `GET /` - Main web interface
- `POST /predict` - Image prediction endpoint
- `GET /health` - Health check endpoint

### Dataset
**Sign Language MNIST** (https://www.kaggle.com/datamunge/sign-language-mnist)
- Training samples: 27,455 images
- Test samples: 7,172 images
- Format: 28×28 pixel values with labels 0-24 (A-Y)
- **Note:** J (index 9) and Z (index 25) excluded due to motion requirements
- Each image represents a single ASL letter gesture

## 🛠️ Features

### Core Functionality
- **Real-time Detection:** Live webcam feed with instant gesture recognition
- **High Accuracy:** CNN model trained on Sign Language MNIST dataset
- **Web Interface:** Modern FastAPI-based web application
- **Cross-platform:** Runs on Windows, macOS, and Linux

### Advanced Features
- **Confidence Scoring:** Displays prediction confidence for each gesture
- **Health Monitoring:** Built-in health check endpoint for system monitoring
- **Error Handling:** Robust error handling and logging
- **GPU/CPU Flexibility:** Automatic fallback to CPU if GPU issues occur

![steps](https://user-images.githubusercontent.com/34855465/76790048-1625c580-67e4-11ea-9fcb-77339e2c4658.png)

### Future Enhancements
- Text autocompletion and word suggestions
- Sentence building capabilities
- Multi-language support 


## 🎯 Use Cases & Applications

### Educational
- **Inclusive Classrooms:** Enable deaf students to participate actively in discussions
- **ASL Learning:** Interactive tool for learning American Sign Language
- **Teacher Training:** Help educators understand ASL communication

### Professional
- **Interpretation Services:** Supplement professional interpreters in various settings
- **Customer Service:** Improve accessibility in retail and service industries
- **Healthcare:** Assist medical professionals in communicating with deaf patients

### Personal & Social
- **Family Communication:** Bridge communication gaps in families with deaf members
- **Tourism:** Help travelers communicate in deaf communities
- **Technology Integration:** Embed in smart devices for voice-free interaction

## 🔧 Development

### Project Structure
```
├── app.py              # FastAPI web application
├── main.py             # Standalone OpenCV application
├── model.py            # PyTorch model definition (legacy)
├── sign_mnist.h5       # Trained TensorFlow model
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
├── Dataset/           # Training and test data
└── AI Experiments/    # Jupyter notebooks for development
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Troubleshooting
- **GPU Issues:** The app automatically falls back to CPU if CUDA errors occur
- **Camera Access:** Ensure your webcam is not being used by other applications
- **Dependencies:** Run `pip install -r requirements.txt` to install all required packages

import os
import requests

# Check and download model if not present
def download_model_from_drive():
    model_path = "model.onnx"
    if not os.path.exists(model_path):
        print("Downloading model.onnx from Google Drive...")
        file_id = "18bVmR5rI9rgDg_cFnr9HSVpRZxYQu6Y8"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("Model downloaded successfully.")

download_model_from_drive()
from flask import Flask, request, render_template
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

app = Flask(__name__)

# Ensure directories exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/charts", exist_ok=True)

# Load ONNX model
model_path = "model.onnx"
try:
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
except Exception as e:
    print(f"Error loading model: {str(e)}")
    session = None

# Class labels
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust', 'Corn___healthy', 'Corn___Northern_Leaf_Blight',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# Store feedback in memory (use a database in production)
feedback_list = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not session:
        return "Model not loaded", 500

    if 'file' not in request.files:
        return "No image uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    try:
        # Save uploaded image
        filename = f"{uuid.uuid4()}.jpg"
        image_path = os.path.join("static/uploads", filename)
        file.save(image_path)

        # Preprocess image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run inference
        outputs = session.run(None, {input_name: img_array})[0][0]
        top_5_indices = outputs.argsort()[-5:][::-1]
        top_5_labels = [class_labels[i] for i in top_5_indices]
        top_5_probs = [float(outputs[i]) for i in top_5_indices]

        # Pair labels and probabilities
        top_5_predictions = list(zip(top_5_labels, top_5_probs))

        # Create bar chart
        chart_filename = f"chart_{uuid.uuid4()}.png"
        chart_path = os.path.join("static/charts", chart_filename)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=top_5_probs, y=top_5_labels, palette="viridis")
        plt.xlabel("Confidence")
        plt.title("Top 5 Predictions")
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()

        return render_template(
            'result.html',
            prediction=top_5_labels[0],
            chart_path=chart_path,
            image_path=image_path,
            top_5_predictions=top_5_predictions
        )
    except Exception as e:
        return f"Error processing image: {str(e)}", 500

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        email = request.form.get('email')
        phone = request.form.get('phone')
        message = request.form.get('message')
        if email and phone and message:
            feedback_list.append({'email': email, 'phone': phone, 'message': message})
            return render_template('feedback.html', success=True, feedback_list=feedback_list)
        return render_template('feedback.html', error="Please fill all fields", feedback_list=feedback_list)
    return render_template('feedback.html', feedback_list=feedback_list)

if __name__ == '__main__':
    app.run(debug=True)

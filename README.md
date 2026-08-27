# 🌱 AI Soil Intelligence

### 🤖 AI-Powered Soil Analysis & Soil pH Prediction System

AI Soil Intelligence is a machine learning-based application designed to analyze soil and predict **soil pH** using soil image features. The system extracts **RGB and HSV color features** from a soil image and uses a trained machine learning model to predict the soil pH level and classify the soil condition.

---

## 🚀 Project Overview

Soil pH is one of the most important factors affecting plant growth and nutrient availability.

Traditional soil testing generally requires laboratory equipment and can take time. This project explores an AI-based approach where a soil image is processed to extract visual color characteristics and a machine learning model predicts the approximate soil pH.

### 🔄 Workflow

```text
📷 Soil Image
      ↓
🖼️ Image Processing
      ↓
🎨 RGB & HSV Feature Extraction
      ↓
🤖 Trained ML Model
      ↓
📊 Soil pH Prediction
      ↓
🌱 Soil Classification
```

---

## ✨ Features

* 📷 Upload a soil image
* 🎨 Extract RGB color features
* 🌈 Extract HSV color features
* 🤖 Machine Learning-based pH prediction
* 📊 Soil pH classification
* ⚡ Fast prediction
* 🌐 API support using FastAPI
* 🖥️ Application interface
* 💾 Pre-trained `.pkl` model
* 📁 CSV dataset for model development

---

## 🧠 Machine Learning

The project uses soil image features to predict soil pH.

### Features

The model uses the following image-based features:

| Feature | Description           |
| ------- | --------------------- |
| R       | Average Red channel   |
| G       | Average Green channel |
| B       | Average Blue channel  |
| H       | Hue                   |
| S       | Saturation            |
| V       | Value                 |

### Target

```text
Soil pH
```

The trained model is stored as:

```text
soil_ph_best_model.pkl
```

---

## 🌱 Soil pH Classification

The predicted pH value can be interpreted approximately as:

|  pH Range | Soil Classification    |
| --------: | ---------------------- |
|     < 4.5 | Extremely Acidic       |
| 4.5 – 5.0 | Strongly Acidic        |
| 5.1 – 5.5 | Moderately Acidic      |
| 5.6 – 6.0 | Slightly Acidic        |
| 6.1 – 7.0 | Neutral / Near Neutral |
| 7.1 – 7.5 | Slightly Alkaline      |
| 7.6 – 8.5 | Moderately Alkaline    |
|     > 8.5 | Strongly Alkaline      |

> **Note:** Image-based pH prediction is an experimental AI approach and should not replace laboratory soil testing for agricultural decisions.

---

## 🛠️ Technologies Used

### Programming

* Python

### Machine Learning

* Scikit-learn
* NumPy
* Pandas

### Image Processing

* OpenCV
* Pillow

### Backend / API

* FastAPI
* Uvicorn
* Flask

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

## 📂 Project Structure

```text
AI-Soil-Intelligence/
│
├── 📄 app.py
├── 📄 main.py
├── 📓 model.ipynb
│
├── 🤖 soil_ph_best_model.pkl
├── 📊 soil_ph_dataset.csv
│
├── 🖼️ Logo.png
├── 🖼️ temp_upload.jpg
│
├── 📁 svg/
│
└── 📄 README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/amit-singh15/AI-Soil-Intelligence.git
```

### 2. Navigate to the project

```bash
cd AI-Soil-Intelligence
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install numpy pandas scikit-learn opencv-python pillow fastapi uvicorn flask python-multipart
```

---

# ▶️ Run the Application

## FastAPI

Run:

```bash
uvicorn main:app --reload
```

The API will start at:

```text
http://127.0.0.1:8000
```

Open the Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 🖥️ Run Flask Application

If `app.py` contains the Flask application:

```bash
python app.py
```

The application will normally be available at:

```text
http://127.0.0.1:5000
```

---

# 🔌 API Workflow

The API can be used to send a soil image to the backend.

```text
POST /predict
```

### Request

```text
Soil Image
```

### Processing

```text
Image
 ↓
OpenCV
 ↓
RGB Extraction
 ↓
HSV Extraction
 ↓
Feature Vector
 ↓
soil_ph_best_model.pkl
```

### Response

Example:

```json
{
    "predicted_ph": 6.45,
    "classification": "Slightly Acidic"
}
```

---

# 🤖 Model Loading

The trained model can be loaded using Python:

```python
import pickle

with open("soil_ph_best_model.pkl", "rb") as file:
    model = pickle.load(file)
```

Then the extracted image features can be passed to the model:

```python
prediction = model.predict(features)
```

---

# 📊 Dataset

The project includes:

```text
soil_ph_dataset.csv
```

The dataset contains soil image/color-related features used for machine learning development and experimentation.

Example features:

```text
R
G
B
H
S
V
pH
```

---

# 🧪 Example Prediction

Example extracted features:

```text
R: 109.42
G: 84.89
B: 66.23
H: 12.82
S: 113.58
V: 109.42
```

Example prediction:

```text
Predicted Soil pH: 4.75

Classification:
Strongly Acidic
```

---

# 📓 Model Development

The complete model development process is available in:

```text
model.ipynb
```

The notebook covers:

1. Dataset loading
2. Data preprocessing
3. Feature selection
4. Model training
5. Model evaluation
6. Model comparison
7. Best model selection
8. Model serialization using Pickle

---

# 🔮 Future Improvements

The project can be extended with additional soil parameters:

* 🧪 Nitrogen (N)
* 🧪 Phosphorus (P)
* 🧪 Potassium (K)
* 💧 Moisture
* 🌡️ Temperature
* ⚡ Electrical Conductivity (EC)
* 🌿 Organic Carbon
* 🧪 Soil pH

Future versions can combine:

```text
📷 Soil Image
       +
🧪 Soil Sensor Data
       ↓
🤖 AI Soil Analysis
       ↓
🌱 Soil Health Prediction
       ↓
🌾 Crop Recommendation
       ↓
💡 Fertilizer Recommendation
```

---

# 🎯 Applications

AI Soil Intelligence can potentially be used for:

* 🌾 Smart Agriculture
* 👨‍🌾 Precision Farming
* 🌱 Soil Health Monitoring
* 🌿 Crop Planning
* 🧪 Soil Research
* 📊 Agricultural Data Analysis
* 🤖 AI-based Farming Solutions

---

# ⚠️ Disclaimer

This project is intended for **educational, research, and prototype purposes**.

Image-based soil pH prediction can be affected by:

* Lighting conditions
* Camera quality
* Soil moisture
* Soil texture
* Image background
* Camera calibration
* Soil composition

For accurate agricultural decisions, results should be verified using **laboratory soil testing or calibrated soil sensors**.

---

# 👨‍💻 Author

## Amit Singh

**B.Tech – Computer Science & Engineering (AI)**

### 🔗 Connect with me

* GitHub: https://github.com/amit-singh15
* Project: https://github.com/amit-singh15/AI-Soil-Intelligence

---

# ⭐ Support

If you find this project useful, please consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is available for educational and research purposes.

# 🍷 Wine Quality Classification

## 📌 Project Overview

Wine Quality Classification is a Machine Learning project that predicts whether a wine is **Good or Bad** based on its chemical properties.

The project uses **Supervised Machine Learning** and provides a web interface where users can enter wine properties and get the prediction.

## 🎯 Objective

The main objective is to classify wine quality using Machine Learning algorithms based on different chemical properties.

## 📊 Dataset

The Wine Quality dataset contains chemical properties of red wine.

### Features Used
- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

### Target
- **Good Wine**
- **Bad Wine**

The original quality values were converted into two classes:
- Quality ≤ 5 → Bad Wine
- Quality > 5 → Good Wine

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Duplicate Removal
   ↓
Feature & Target Separation
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Random Forest
   ↓
Flask API
   ↓
Frontend
   ↓
Good Wine / Bad Wine Prediction
```

## 🤖 Machine Learning Algorithms

The following algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Model Accuracy

| Model | Accuracy |
|---|---:|
| Logistic Regression | 73.53% |
| Decision Tree | 62.50% |
| Random Forest | 76.84% |

Random Forest was used for the final prediction based on the tested model accuracy.

## 🛠️ Technologies Used

### Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Backend
- Flask
- Flask-CORS

### Frontend
- HTML
- CSS
- JavaScript

### Tools
- VS Code
- Git
- GitHub

## 🌐 Frontend and Backend

The frontend is developed using HTML, CSS, and JavaScript.

The backend is developed using Flask. The frontend sends the wine properties to the Flask API, and the trained Random Forest model predicts the wine quality.

### Prediction Flow

```text
User enters wine properties
        ↓
Frontend
        ↓
Flask API
        ↓
StandardScaler
        ↓
Random Forest Model
        ↓
Prediction
        ↓
Good Wine / Bad Wine
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Install required packages

```bash
pip install pandas numpy scikit-learn joblib flask flask-cors requests
```

### 3. Train the model

From the project root:

```bash
python train_model.py
```

This creates the trained model and scaler inside the `model` folder.

### 4. Start the Flask backend

```bash
cd backend
python app.py
```

The backend will run at:

```text
http://127.0.0.1:5000
```

### 5. Run the frontend

Open `frontend/index.html` using **Live Server** in VS Code.

Then enter the wine properties and click **Predict Wine Quality**.

## 📁 Project Structure

```text
Wine_Quality_Classification/
│
├── dataset/
│   └── winequality-red.csv
│
├── model/
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── backend/
│   └── app.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── train_model.py
├── test_api.py
└── README.md
```

## 👩‍💻 Project Developed By

**Sangavi S.**

Final Year BCA Student  
Bishop Heber College, Trichy

## 📌 Conclusion

This project demonstrates how Machine Learning can be integrated with a web application to classify wine quality based on its chemical properties. It combines **Machine Learning, Flask backend, API integration, and frontend development** in a single project.
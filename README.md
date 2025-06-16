# Chronic Kidney Disease Detection using Recurrent Neural Network

This project builds a neural network model to predict chronic kidney disease using a dataset of clinical attributes. The model is built using TensorFlow's Keras API and evaluates classification accuracy through preprocessing, training, and evaluation phases.

## 📁 Dataset

The dataset used is `chronic_kidney_disease.csv`, which contains clinical data including:
- Numeric attributes (e.g., age, blood pressure, blood urea, serum creatinine, etc.)
- Categorical attributes (e.g., red blood cells, pus cell, bacteria, etc.)
- Target variable: `classification` (ckd or notckd)

## 🛠️ Technologies Used

- Python 3.x
- pandas
- numpy
- scikit-learn
- TensorFlow (Keras)
- matplotlib

## ⚙️ Model Architecture

A simple feed-forward neural network:
- Input Layer: Dense layer with 64 units and ReLU activation
- Output Layer: Dense layer with 1 unit and Sigmoid activation

## 🔄 Workflow

1. **Data Cleaning**
   - Removed non-numeric columns
   - Handled missing values by dropping rows with `NaN`
   
2. **Preprocessing**
   - One-hot encoded selected categorical variables (`rbc`, `pc`, `pcc`, `ba`)
   - Dropped additional non-numeric or irrelevant columns
   - Scaled features using `StandardScaler`
   - Encoded the target variable using `LabelEncoder`
   
3. **Train-Test Split**
   - Split data into 80% training and 20% testing

4. **Model Training**
   - Binary classification using `binary_crossentropy` loss
   - Optimizer: Adam
   - Trained over 8 epochs with batch size of 32

5. **Evaluation**
   - Printed training and testing accuracy
   - Visualized training vs validation accuracy and loss

## 📊 Results

- **Accuracy:**  0.96875
- **Loss:**   0.3227



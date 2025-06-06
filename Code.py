import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import matplotlib.pyplot as plt

data = pd.read_csv('/content/chronic_kidney_disease.csv')

non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
data = data.drop(columns=non_numeric_cols)

data.isnull().any()

data = data.dropna()

data.isnull().any()

print(data)

# Load your dataset
# Assuming 'data.csv' is the name of your dataset file
data = pd.read_csv('/content/chronic_kidney_disease.csv')

# Preprocess the data
data = data.dropna()

# Extract categorical columns
categorical_cols = ['rbc', 'pc', 'pcc', 'ba']

# Check if categorical columns exist in the dataset
missing_cols = set(categorical_cols) - set(data.columns)
if missing_cols:
    raise ValueError(f"Missing categorical columns: {missing_cols}")

# Apply one-hot encoding to categorical columns
data_encoded = pd.get_dummies(data, columns=categorical_cols)

# Check if there are still non-numeric values in the dataset
non_numeric_cols = data_encoded.select_dtypes(exclude=[np.number]).columns
data_encoded = data_encoded.drop(columns=['wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'])

# Encode the target variable
label_encoder = LabelEncoder()
data_encoded['classification'] = label_encoder.fit_transform(data_encoded['classification'])

# Split the dataset into features and labels
X = data_encoded.drop(columns=['classification'])
y = data_encoded['classification']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the RNN model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=8, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the training data
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
print(f'Training Accuracy: {train_accuracy}')

# Evaluate the model on the testing data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Testing Accuracy: {test_accuracy}')

# Predict on the test data
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate accuracy using sklearn's accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# Plot training & validation accuracy values
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

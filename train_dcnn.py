import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# 1. GENERATE DATA WITH 3-TIER DRUG RISK LOGIC
def generate_drug_risk_data(samples=3000):
    np.random.seed(42)
    data = []
    labels = []
    
    for _ in range(samples):
        age = np.random.randint(18, 90)
        gender = np.random.randint(0, 2)
        ecg = np.random.randint(0, 2) 
        echo = np.random.randint(30, 75) 
        fbs = np.random.randint(70, 280) 
        bp_sys = np.random.randint(90, 210) 
        hba1c = np.random.uniform(4.0, 13.0)
        
        # DRUG RISK LOGIC:
        # 0: Low (Green), 1: Medium (Yellow), 2: High (Red)
        score = 0
        if fbs > 180 or hba1c > 9.0: score += 3
        if bp_sys > 160: score += 2
        if echo < 40: score += 3
        if age > 65 and score > 1: score += 1

        if score <= 2:
            risk = 0 # Low Risk
        elif score <= 4:
            risk = 1 # Medium Risk
        else:
            risk = 2 # High Risk
        
        features = [age, gender, ecg, echo, 140, 14, fbs, hba1c, bp_sys, 1.0, 140, 25]
        data.append(features)
        labels.append(risk)
        
    return np.array(data), np.array(labels)

X, y = generate_drug_risk_data()

# 2. PREPROCESSING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

X_final = np.expand_dims(X_res, axis=2)
y_final = to_categorical(y_res, num_classes=3) # Changed to 3

# 3. DCNN ARCHITECTURE FOR 3 CLASSES
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2)

model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=(12, 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    
    Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax') # 3 Output Neurons for Low/Medium/High
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. TRAIN
model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test))

# 5. SAVE
model.save("drug_risk_model.h5")
print("SUCCESS: 3-Tier Drug Risk Model Saved.")
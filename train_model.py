import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# 1. Configuration
DATASET_PATH = 'dataset'
MODEL_DIR = 'model'
IMG_SIZE = 128
CLASSES = [
    "01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb", 
    "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"
]

# Ensure directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Data Loading function
def load_data():
    X, y = [], []
    print("Loading and preprocessing images... This may take a few minutes.")
    
    for subject_folder in os.listdir(DATASET_PATH):
        subject_path = os.path.join(DATASET_PATH, subject_folder)
        if not os.path.isdir(subject_path):
            continue
            
        for gesture_folder in os.listdir(subject_path):
            gesture_path = os.path.join(subject_path, gesture_folder)
            if not os.path.isdir(gesture_path) or gesture_folder not in CLASSES:
                continue
                
            label = CLASSES.index(gesture_folder)
            
            for img_name in os.listdir(gesture_path):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(gesture_path, img_name)
                    
                    img = Image.open(img_path).convert('L')
                    img = img.resize((IMG_SIZE, IMG_SIZE))
                    
                    X.append(np.array(img))
                    y.append(label)
                    
    X = np.array(X, dtype="float32") / 255.0
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = to_categorical(np.array(y), num_classes=len(CLASSES))
    
    return X, y

X, y = load_data()
print(f"Total images loaded: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build the CNN Model
print("Building the CNN model...")
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train the Model
print("Starting training...")
history = model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# 5. Evaluate and Save
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

model_path = os.path.join(MODEL_DIR, 'gesture_model.keras')
model.save(model_path)
print(f"Model saved successfully to {model_path}!")
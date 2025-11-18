# train.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam 

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mnist_cnn.h5")

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension
    x_train = x_train[..., np.newaxis]   # (N, 28, 28, 1)
    x_test = x_test[..., np.newaxis]

    # One-hot encode labels for categorical cross-entropy
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    return (x_train, y_train_cat), (x_test, y_test_cat)

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_data()

    print("Building model...")
    model = build_model()
    model.summary()

    print("Training model...")
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    print(f"Saving model to {MODEL_PATH} ...")
    model.save(MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    main()

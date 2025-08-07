import os
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models


# 1. CNN Model Definition and Training
def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_and_save_model(model_path='digit_cnn.h5'):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    model = build_cnn_model()
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {acc:.4f}")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model


def load_model(model_path='digit_cnn.h5'):
    if not os.path.exists(model_path):
        return train_and_save_model(model_path)
    print(f"Loading model from {model_path}")
    return tf.keras.models.load_model(model_path)


# 2. OCR and Digit Recognition Function
def ocr_and_digit_recognize(image_path, model):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Step 1: Display original image
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.subplot(2, 2, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')

    # Step 3: Thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    plt.subplot(2, 2, 3)
    plt.imshow(thresh, cmap='gray')
    plt.title("Thresholded Image")
    plt.axis('off')

    # Step 4: OCR with Tesseract
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(thresh, config=custom_config)
    print("\nExtracted Text (OCR):\n", extracted_text)

    # Step 5: CNN Digit Recognition
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = image.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 100:  # Skip small regions (likely noise)
            continue
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (28, 28))
        roi = roi / 255.0
        roi = roi.reshape(1, 28, 28, 1)

        pred = model.predict(roi)
        digit = np.argmax(pred)

        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(annotated, str(digit), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # Step 6: Annotated Output
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title("Digit Recognition Output")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return extracted_text, annotated


# 3. Main Execution
if __name__ == "__main__":
    model = load_model('digit_cnn.h5')
    image_path = r"D:\CODING\check.png"  # Update this to your image path
    extracted_text, annotated_image = ocr_and_digit_recognize(image_path, model)
    print("\n<<< OCR Result >>>\n", extracted_text)

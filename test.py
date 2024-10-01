import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """Load an image and preprocess it for the model.

    Args:
        image_path (str): Path to the image.

    Returns:
        np.ndarray: Preprocessed image ready for inference.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(512, 512))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

def run_inference(model_path: str, test_image_path: str) -> np.ndarray:
    """Run inference on a test image using the trained model.

    Args:
        model_path (str): Path to the trained model.
        test_image_path (str): Path to the test image.

    Returns:
        np.ndarray: Output of the model (super-resolved image).
    """
    model = load_model(model_path)
    image = load_and_preprocess_image(test_image_path)
    prediction = model.predict(image)
    return prediction

def display_images(original: np.ndarray, super_resolved: np.ndarray) -> None:
    """Display the original and super-resolved images.

    Args:
        original (np.ndarray): Original image.
        super_resolved (np.ndarray): Super-resolved image.
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original[0])  # Remove batch dimension
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(super_resolved[0])  # Remove batch dimension
    plt.title("Super-Resolved Image")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    model_path = 'path/to/trained_model.h5'  # Update with actual path
    test_image_path = 'path/to/test_image.png'  # Update with actual path

    # Run inference
    super_resolved_image = run_inference(model_path, test_image_path)
    
    # Load the original image for comparison
    original_image = load_and_preprocess_image(test_image_path)

    # Display the results
    display_images(original_image, super_resolved_image)

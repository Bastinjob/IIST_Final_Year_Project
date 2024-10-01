import tensorflow as tf
from tensorflow import keras

def train_models(degradation_encoder: keras.Model, generator: keras.Model, train_path: str, test_path: str, priming_epochs: int = 50, epochs: int = 300) -> None:
    """
    Trains the degradation encoder and generator models.

    Args:
        degradation_encoder (keras.Model): The degradation encoder model.
        generator (keras.Model): The super resolution generator model.
        train_path (str): Path to training images.
        test_path (str): Path to test images.
        priming_epochs (int): Number of epochs to prime the degradation encoder.
        epochs (int): Total number of training epochs.
    """
    train_datagen = keras.preprocessing.image.ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(train_path, target_size=(512, 512), batch_size=32)
    
    test_datagen = keras.preprocessing.image.ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(test_path, target_size=(512, 512), batch_size=32)

    for epoch in range(epochs):
        if epoch < priming_epochs:
            degradation_encoder.fit(train_generator, epochs=1, validation_data=test_generator)
        else:
            degradation_encoder.fit(train_generator, epochs=1, validation_data=test_generator)
            generator.fit(train_generator, epochs=1, validation_data=test_generator)

if __name__ == "__main__":
    # Example usage
    from model import build_degradation_encoder, build_generator
    degradation_encoder = build_degradation_encoder()
    generator = build_generator()
    
     # Assume paths are set for training and test datasets
    train_path = '/path/to/train_data'
    test_path = '/path/to/test_data'
    
    train_models(degradation_encoder, generator, train_path, test_path)
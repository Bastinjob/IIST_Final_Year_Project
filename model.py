import tensorflow as tf
from tensorflow.keras import layers, models

def contra_loss(zi: tf.Tensor, zj: tf.Tensor, tau: float = 1.0) -> tf.Tensor:
    """
    Computes the contrastive loss between two tensors.

    Args:
        zi (tf.Tensor): First tensor.
        zj (tf.Tensor): Second tensor.
        tau (float): Temperature parameter for scaling.

    Returns:
        tf.Tensor: Computed contrastive loss.
    """
    z = tf.concat((zi, zj), 0)
    loss = 0.0
    for k in range(zi.shape[0]):
        i = k
        j = k + zi.shape[0]

        # Cosine similarity
        cosine_sim = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.NONE)
        sim = -cosine_sim(tf.reshape(z[i], (1, -1)), tf.reshape(z[j], (1, -1)))
        numerator = tf.exp(sim / tau)

        # Denominator calculations
        sim_ik = -cosine_sim(tf.reshape(z[i], (1, -1)), z[tf.range(z.shape[0]) != i])
        sim_jk = -cosine_sim(tf.reshape(z[j], (1, -1)), z[tf.range(z.shape[0]) != j])
        denominator_ik = tf.reduce_sum(tf.exp(sim_ik / tau))
        denominator_jk = tf.reduce_sum(tf.exp(sim_jk / tau))

        # Individual losses
        loss_ij = -tf.math.log(numerator / denominator_ik)
        loss_ji = -tf.math.log(numerator / denominator_jk)
        loss += loss_ij + loss_ji

    return loss / z.shape[0]

def build_degradation_encoder() -> models.Model:
    """
    Builds the degradation encoder model.
    
    Returns:
        tf.keras.Model: Degradation encoder model.
    """
    encoder = models.Sequential([
        layers.Conv2D(64, kernel_size=3, input_shape=(512, 512, 3)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.1),
        layers.Conv2D(128, kernel_size=3),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.1),
        layers.Conv2D(256, kernel_size=3),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.1),
        layers.AveragePooling2D(1),
    ])

    projection_head = models.Sequential([
        layers.Flatten(),
        layers.Dense(256),
        layers.LeakyReLU(0.1),
        layers.Dense(256),
    ])

    degradation_encoder = models.Sequential([encoder, projection_head])
    degradation_encoder.compile(loss=contra_loss, optimizer='adam')

    return degradation_encoder

def build_generator() -> models.Model:
    """
    Builds the super resolution generator model.
    
    Returns:
        tf.keras.Model: Super resolution generator model.
    """
    input1 = layers.Input(shape=(512, 512, 1))
    input2 = layers.Input(shape=(None, 1))

    # Resizing the degradation representation
    resized_deg_rep = layers.Dense(64)(input2)
    resized_deg_rep = layers.LeakyReLU(0.1)(resized_deg_rep)
    resized_deg_rep = layers.Dense(64)(resized_deg_rep)

    # Convolutions on input image
    feat = layers.Conv2D(64, kernel_size=3)(input1)
    feat = layers.LeakyReLU(0.1)(feat)
    feat = layers.Conv2D(64, kernel_size=3)(feat)
    feat = layers.LeakyReLU(0.1)(feat)

    out = feat + resized_deg_rep

    # More convolutional layers
    for _ in range(3):
        out = layers.Conv2D(64, kernel_size=3)(out)
        out = layers.LeakyReLU(0.1)(out)
        out = layers.Conv2D(64, kernel_size=3)(out)
        out = layers.LeakyReLU(0.1)(out)
        out = out + resized_deg_rep

    # Upsampling
    out = layers.Flatten()(out)
    out = layers.Dense(512 * 512)(out)

    generator = models.Model(inputs=[input1, input2], outputs=out)
    generator.compile(loss='mean_squared_error')  # You may choose a suitable loss

    return generator

if __name__ == "__main__":
    # Example usage
    degradation_encoder = build_degradation_encoder()
    generator = build_generator()
    print("Models built successfully!")

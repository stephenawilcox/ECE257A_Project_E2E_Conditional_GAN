from __future__ import division
import numpy as np
import tensorflow as tf
import os

''' This file implements frequency-selective fading channel using conditional GAN '''

os.environ["CUDA_DEVICE_ORDER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

channel_data_path = 'data_channel_quadriga'

def load_channel_data():
    # Load dimensions
    with open(f'{channel_data_path}/dimensions.txt', 'r') as f:
        dim1, dim2 = map(int, f.read().split(','))

    # Load channel data
    cn_real = np.fromfile(f'{channel_data_path}/channel_real.bin', dtype=np.float64)
    cn_real = cn_real.reshape(dim1, dim2)
    cn_imag = np.fromfile(f'{channel_data_path}/channel_imag.bin', dtype=np.float64)
    cn_imag = cn_imag.reshape(dim1, dim2)

    return cn_real, cn_imag



def generator_conditional(z, conditioning):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        # Combine noise and conditioning (channel info for all subcarriers)
        z_combine = tf.concat([z, conditioning], -1)

        # Expanded network to handle frequency-selective features
        conv1_g = tf.layers.conv1d(inputs=z_combine, filters=512, kernel_size=5, padding='same')
        conv1_g = tf.nn.leaky_relu(conv1_g)

        conv2_g = tf.layers.conv1d(inputs=conv1_g, filters=256, kernel_size=3, padding='same')
        conv2_g = tf.nn.leaky_relu(conv2_g)

        # Add residual connections for better feature preservation
        conv3_g = tf.layers.conv1d(inputs=conv2_g, filters=128, kernel_size=3, padding='same')
        conv3_g = tf.nn.leaky_relu(conv3_g)
        conv3_g = tf.add(conv3_g, tf.layers.conv1d(conv2_g, filters=128, kernel_size=1, padding='same'))

        conv4_g = tf.layers.conv1d(inputs=conv3_g, filters=2 * num_subcarriers, kernel_size=3, padding='same')
        # Reshape output to separate subcarriers
        output_shape = tf.shape(conv4_g)
        return tf.reshape(conv4_g, [output_shape[0], num_subcarriers, 2])


def discriminator_conditional(x, conditioning):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        # Combine input and conditioning
        z_combine = tf.concat([x, conditioning], -1)

        conv1 = tf.layers.conv1d(inputs=z_combine, filters=512, kernel_size=5, padding='same')
        conv1 = tf.nn.leaky_relu(conv1)

        conv2 = tf.layers.conv1d(inputs=conv1, filters=256, kernel_size=3, padding='same')
        conv2 = tf.nn.leaky_relu(conv2)

        # Add attention mechanism for frequency-selective features
        attention = tf.layers.dense(conv2, 256, activation=tf.nn.tanh)
        attention = tf.layers.dense(attention, 1, activation=tf.nn.sigmoid)
        conv2_attended = tf.multiply(conv2, attention)

        conv3 = tf.layers.conv1d(inputs=conv2_attended, filters=128, kernel_size=3, padding='same')
        conv3 = tf.nn.leaky_relu(conv3)

        # Global average pooling
        pooled = tf.reduce_mean(conv3, axis=1)

        FC = tf.layers.dense(pooled, 100, activation=tf.nn.leaky_relu)
        D_logit = tf.layers.dense(FC, 1, activation=None)
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit


def frequency_selective_channel(input_layer, h_real, h_imag, std):
    """
    Implements frequency-selective fading channel
    Args:
        input_layer: Input signal [batch_size, num_subcarriers, 2]
        h_real: Real channel coefficients [batch_size, num_subcarriers, 1]
        h_imag: Imaginary channel coefficients [batch_size, num_subcarriers, 1]
        std: Noise standard deviation
    """
    # Convert to complex numbers
    h_complex = tf.complex(real=h_real, imag=h_imag)
    input_complex = tf.complex(real=input_layer[:, :, 0], imag=input_layer[:, :, 1])

    # Add frequency-selective fading per subcarrier
    noise = tf.complex(
        real=tf.random_normal(shape=tf.shape(input_complex), mean=0.0, stddev=std, dtype=tf.float32),
        imag=tf.random_normal(shape=tf.shape(input_complex), mean=0.0, stddev=std, dtype=tf.float32)
    )

    # Apply channel effect per subcarrier
    output_complex = tf.add(tf.multiply(h_complex, input_complex), noise)

    # Convert back to real representation
    return tf.stack([tf.real(output_complex), tf.imag(output_complex)], axis=-1)


def generate_channel_coefficients(batch_size, num_subcarriers, coherence_bandwidth):
    """
    Generate correlated channel coefficients for frequency-selective fading
    """
    # Generate base coefficients
    h_base_real = np.random.normal(0, 1 / np.sqrt(2), [batch_size, int(num_subcarriers / coherence_bandwidth), 1])
    h_base_imag = np.random.normal(0, 1 / np.sqrt(2), [batch_size, int(num_subcarriers / coherence_bandwidth), 1])

    # Interpolate to get correlated coefficients for all subcarriers
    h_real = np.repeat(h_base_real, coherence_bandwidth, axis=1)
    h_imag = np.repeat(h_base_imag, coherence_bandwidth, axis=1)

    # Add small random variations
    h_real += np.random.normal(0, 0.1, [batch_size, num_subcarriers, 1])
    h_imag += np.random.normal(0, 0.1, [batch_size, num_subcarriers, 1])

    return h_real, h_imag


# Parameters for frequency-selective fading
num_subcarriers = 64  # Number of subcarriers
coherence_bandwidth = 4  # Number of adjacent subcarriers with similar channel conditions
batch_size = 300
block_length = num_subcarriers
Z_dim_c = 16
learning_rate = 1e-4

# Placeholders
X = tf.placeholder(tf.float32, shape=[None, num_subcarriers, 2])
Z = tf.placeholder(tf.float32, shape=[None, block_length, Z_dim_c])
h_r = tf.placeholder(tf.float32, shape=[None, num_subcarriers, 1])
h_i = tf.placeholder(tf.float32, shape=[None, num_subcarriers, 1])
Noise_std = tf.placeholder(tf.float32, shape=[])

# Channel information for conditioning
Channel_info = tf.concat([h_r, h_i], axis=-1)
Conditions = tf.concat([X, Channel_info], axis=-1)

# Generate fake samples
G_sample = generator_conditional(Z, Conditions)

# Real channel samples
R_sample = frequency_selective_channel(X, h_r, h_i, Noise_std)

# Discriminator outputs
D_prob_real, D_logit_real = discriminator_conditional(R_sample, Conditions)
D_prob_fake, D_logit_fake = discriminator_conditional(G_sample, Conditions)

# Loss functions
D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

# Training operations
Disc_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
Gen_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]

D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(D_loss, var_list=Disc_vars)
G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5).minimize(G_loss, var_list=Gen_vars)


# Training loop
def train_model(sess, num_epochs=100):
    for epoch in range(num_epochs):
        # Generate channel coefficients with frequency selectivity
        h_real, h_imag = generate_channel_coefficients(batch_size, num_subcarriers, coherence_bandwidth)

        # Train discriminator
        _, d_loss = sess.run([D_solver, D_loss],
                             feed_dict={X: generate_batch_data(batch_size),
                                        Z: np.random.normal(0, 1, [batch_size, block_length, Z_dim_c]),
                                        h_r: h_real,
                                        h_i: h_imag,
                                        Noise_std: np.sqrt(0.1)})

        # Train generator
        _, g_loss = sess.run([G_solver, G_loss],
                             feed_dict={X: generate_batch_data(batch_size),
                                        Z: np.random.normal(0, 1, [batch_size, block_length, Z_dim_c]),
                                        h_r: h_real,
                                        h_i: h_imag,
                                        Noise_std: np.sqrt(0.1)})

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: D_loss = {d_loss:.4f}, G_loss = {g_loss:.4f}")


def generate_batch_data(batch_size):
    """Generate random OFDM symbols"""
    return np.random.normal(0, 1, [batch_size, num_subcarriers, 2])


# Initialize and train
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    train_model(sess)

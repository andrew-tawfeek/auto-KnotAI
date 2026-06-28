# Knot Mosaic Diffusion Model — TensorFlow Starter Guide

A hands-on reference for building a CNN-based diffusion model that learns local
knotting rules from small mosaics and generates larger ones.

---

## 1. Core TensorFlow Imports

```python
import tensorflow as tf
import numpy as np

# Key modules you'll use throughout:
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
```

Everything below uses these plus standard Python. No exotic dependencies.

---

## 2. One-Hot Encoding Your Mosaics

### The idea

Each mosaic is an n×n grid with entries in {0, 1, ..., 10} (11 tile types).
We convert each entry to a basis vector in R^11.

### Implementation

```python
NUM_TILE_TYPES = 11

def mosaic_to_onehot(mosaic):
    """
    mosaic: np.array of shape (n, n) with integer entries in {0, ..., 10}
    returns: np.array of shape (n, n, 11) — one-hot encoded
    """
    return tf.one_hot(mosaic, depth=NUM_TILE_TYPES).numpy()

def onehot_to_mosaic(onehot):
    """
    Inverse: (n, n, 11) -> (n, n) integer mosaic.
    Takes argmax along the channel axis.
    """
    return tf.argmax(onehot, axis=-1).numpy()

# Example: your 2x2 loop mosaic
mosaic = np.array([[2, 1],
                   [3, 4]])
encoded = mosaic_to_onehot(mosaic)
print(encoded.shape)  # (2, 2, 11)
print(encoded[0, 0])  # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  <- tile 2
```

### Preparing a dataset of mosaics

```python
def prepare_dataset(mosaic_list, batch_size=32):
    """
    mosaic_list: list of np.arrays, each of shape (n, n)
    returns: tf.data.Dataset of one-hot encoded mosaics
    """
    encoded = np.array([mosaic_to_onehot(m) for m in mosaic_list])
    # encoded shape: (num_mosaics, n, n, 11)
    dataset = tf.data.Dataset.from_tensor_slices(encoded.astype(np.float32))
    dataset = dataset.shuffle(buffer_size=len(mosaic_list))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
```

You'll generate `mosaic_list` from your KnotMosaics SageMath package —
enumerate valid 3×3 mosaics and export them as integer arrays.

---

## 3. Building a CNN (U-Net Style)

### Why U-Net?

The denoiser in a diffusion model needs to take in a noisy (n, n, 11) tensor
and output a prediction of the same shape. A U-Net does this via:
- **Encoder**: Conv2D layers that shrink spatial resolution, growing channels
- **Decoder**: UpSampling layers that restore resolution
- **Skip connections**: Concatenate encoder features to decoder features

This gives both local detail (from skips) and global context (from the bottleneck).

### Key layers to know

```python
# --- Convolution: the local, equivariant operation ---
layers.Conv2D(
    filters=64,        # number of output channels (feature maps)
    kernel_size=3,     # 3x3 filter — sees local neighborhood
    padding='same',    # output has same spatial size as input
    activation='relu'  # familiar from feedforward nets
)
# The filter is a (3, 3, C_in, 64) tensor of learned weights
# where C_in is the number of input channels (11 for your first layer)

# --- Downsampling: shrink spatial dimensions ---
layers.MaxPooling2D(pool_size=2)   # (n, n) -> (n/2, n/2)

# --- Upsampling: restore spatial dimensions ---
layers.UpSampling2D(size=2)        # (n/2, n/2) -> (n, n)

# --- Skip connections: concatenate along channel axis ---
layers.Concatenate(axis=-1)

# --- Final output layer ---
layers.Conv2D(
    filters=NUM_TILE_TYPES,  # 11 output channels, one per tile type
    kernel_size=1,            # 1x1 conv = per-pixel linear map
    padding='same',
    activation=None           # raw logits (or softmax for probabilities)
)
```

### A minimal U-Net for small mosaics

```python
def make_unet(input_shape, time_embed_dim=32):
    """
    input_shape: (n, n, 11) for one-hot encoded mosaics
    Returns a Model that takes [noisy_mosaic, time_embedding] -> predicted noise
    """
    # --- Inputs ---
    x_input = layers.Input(shape=input_shape, name='noisy_mosaic')
    t_input = layers.Input(shape=(time_embed_dim,), name='time_embedding')

    # Broadcast time embedding to spatial dims
    # (batch, time_embed_dim) -> (batch, 1, 1, time_embed_dim) -> (batch, n, n, time_embed_dim)
    t = layers.Reshape((1, 1, time_embed_dim))(t_input)
    t = tf.tile(t, [1, input_shape[0], input_shape[1], 1])

    # Concatenate noisy mosaic with time info
    x = layers.Concatenate()([x_input, t])  # (n, n, 11 + time_embed_dim)

    # --- Encoder ---
    # Block 1
    e1 = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    e1 = layers.Conv2D(64, 3, padding='same', activation='relu')(e1)
    p1 = layers.MaxPooling2D(2)(e1)

    # Block 2 (bottleneck for small mosaics)
    b = layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
    b = layers.Conv2D(128, 3, padding='same', activation='relu')(b)

    # --- Decoder ---
    u1 = layers.UpSampling2D(2)(b)
    u1 = layers.Concatenate()([u1, e1])  # skip connection
    d1 = layers.Conv2D(64, 3, padding='same', activation='relu')(u1)
    d1 = layers.Conv2D(64, 3, padding='same', activation='relu')(d1)

    # --- Output ---
    output = layers.Conv2D(NUM_TILE_TYPES, 1, padding='same')(d1)

    return Model(inputs=[x_input, t_input], outputs=output)

# For 3x3 mosaics during training:
model = make_unet(input_shape=(3, 3, 11))
model.summary()
```

**Note on input size flexibility**: Because the model is fully convolutional
(no Dense layers), you can build a *new* model with `input_shape=(10, 10, 11)`
at generation time and copy the trained weights. The Conv2D weights don't
depend on spatial dimensions — this is equivariance in action.

---

## 4. The Diffusion Process

### Forward process: adding noise

For continuous diffusion on one-hot vectors, we add Gaussian noise directly.
(For a more principled discrete approach, see Section 7.)

```python
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule."""
    return np.linspace(beta_start, beta_end, timesteps).astype(np.float32)

def precompute_diffusion_params(betas):
    """Precompute all the cumulative products needed for sampling."""
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)  # alpha_bar_t
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus': sqrt_one_minus_alphas_cumprod,
    }

T = 200  # number of diffusion timesteps
params = precompute_diffusion_params(linear_beta_schedule(T))

def q_sample(x_0, t, params):
    """
    Forward process: sample x_t given x_0 and timestep t.
    x_0: (batch, n, n, 11) one-hot encoded clean mosaic
    t:   (batch,) integer timesteps
    returns: x_t (noisy), epsilon (the noise that was added)
    """
    noise = tf.random.normal(shape=tf.shape(x_0))
    sqrt_alpha = tf.gather(params['sqrt_alphas_cumprod'], t)
    sqrt_one_minus = tf.gather(params['sqrt_one_minus'], t)

    # Reshape for broadcasting: (batch,) -> (batch, 1, 1, 1)
    sqrt_alpha = tf.reshape(sqrt_alpha, (-1, 1, 1, 1))
    sqrt_one_minus = tf.reshape(sqrt_one_minus, (-1, 1, 1, 1))

    x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
    return x_t, noise
```

### Time embedding

```python
def sinusoidal_embedding(t, dim=32):
    """
    Encode integer timestep t into a continuous vector.
    Same idea as positional encoding in transformers.
    t: (batch,) integer tensor
    returns: (batch, dim) float tensor
    """
    half_dim = dim // 2
    freqs = tf.exp(
        -tf.math.log(10000.0) * tf.range(half_dim, dtype=tf.float32) / half_dim
    )
    args = tf.cast(t, tf.float32)[:, None] * freqs[None, :]
    return tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
```

---

## 5. Training Loop

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_step(model, x_0, params):
    """One training step: predict the noise that was added."""
    batch_size = tf.shape(x_0)[0]

    # Sample random timesteps for each example in the batch
    t = tf.random.uniform((batch_size,), 0, T, dtype=tf.int32)

    # Add noise
    x_t, noise = q_sample(x_0, t, params)

    # Time embedding
    t_embed = sinusoidal_embedding(t, dim=32)

    with tf.GradientTape() as tape:
        predicted_noise = model([x_t, t_embed], training=True)
        loss = tf.reduce_mean(tf.square(noise - predicted_noise))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# --- Main training loop ---
def train(model, dataset, params, epochs=100):
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for batch in dataset:
            loss = train_step(model, batch, params)
            epoch_loss += loss.numpy()
            num_batches += 1
        avg_loss = epoch_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
```

---

## 6. Generation (Reverse Process)

```python
def p_sample(model, x_t, t, params):
    """Single reverse diffusion step: x_t -> x_{t-1}."""
    t_batch = tf.fill((tf.shape(x_t)[0],), t)
    t_embed = sinusoidal_embedding(t_batch, dim=32)

    predicted_noise = model([x_t, t_embed], training=False)

    beta_t = params['betas'][t]
    alpha_t = params['alphas'][t]
    alpha_bar_t = params['alphas_cumprod'][t]

    # Predicted x_0 direction
    coeff1 = 1.0 / tf.sqrt(alpha_t)
    coeff2 = beta_t / tf.sqrt(1.0 - alpha_bar_t)
    mean = coeff1 * (x_t - coeff2 * predicted_noise)

    if t > 0:
        noise = tf.random.normal(shape=tf.shape(x_t))
        sigma = tf.sqrt(beta_t)
        return mean + sigma * noise
    else:
        return mean

def generate(model, params, shape=(1, 10, 10, 11)):
    """
    Generate a mosaic by running the full reverse process.
    shape: (batch_size, height, width, num_tile_types)
    """
    # Start from pure noise
    x = tf.random.normal(shape)

    # Denoise step by step
    for t in reversed(range(T)):
        x = p_sample(model, x, t, params)

    # Convert continuous output to discrete tiles
    mosaic = tf.argmax(x, axis=-1)  # (batch, h, w)
    return mosaic.numpy()

# --- Generate a 10x10 mosaic from a model trained on 3x3 ---
# First, build a new model with the larger input shape
model_10x10 = make_unet(input_shape=(10, 10, 11))

# Copy weights from trained 3x3 model (works because Conv2D weights
# are (kernel_h, kernel_w, C_in, C_out) — no spatial dimension dependency)
model_10x10.set_weights(model.get_weights())

# Generate!
generated_mosaic = generate(model_10x10, params, shape=(1, 10, 10, 11))
print(generated_mosaic[0])  # (10, 10) integer array — your knot mosaic
```

---

## 7. Discrete Diffusion (More Natural for Tiles)

Since tiles are categorical, you may want a *discrete* diffusion process
instead of adding Gaussian noise to one-hot vectors. The idea (from D3PM,
Austin et al. 2021):

**Forward process**: Instead of adding Gaussian noise, randomly flip each
tile to a different type with some probability. At each step, each tile
independently transitions according to a matrix Q_t:

```
Q_t[i, j] = probability of tile i becoming tile j at step t
```

Common choice: uniform noise — with probability beta_t, replace the tile
with a uniformly random tile; with probability 1 - beta_t, keep it.

**Reverse process**: The network predicts the *original clean tile* (not noise)
at each position. Output is (n, n, 11) logits, trained with cross-entropy loss.

```python
def discrete_forward(x_0_int, t, beta_schedule):
    """
    x_0_int: (batch, n, n) integer tile indices
    Randomly corrupt tiles toward uniform distribution.
    """
    beta_t = tf.gather(beta_schedule, t)
    beta_t = tf.reshape(beta_t, (-1, 1, 1))

    # With probability beta_t, replace with random tile
    mask = tf.random.uniform(tf.shape(x_0_int)) < beta_t
    random_tiles = tf.random.uniform(
        tf.shape(x_0_int), 0, NUM_TILE_TYPES, dtype=tf.int32
    )
    x_t = tf.where(mask, random_tiles, x_0_int)
    return x_t

# Loss: cross-entropy between predicted tile probabilities and true x_0
def discrete_loss(model, x_0_int, t, beta_schedule):
    x_t = discrete_forward(x_0_int, t, beta_schedule)
    x_t_onehot = tf.one_hot(x_t, NUM_TILE_TYPES)
    t_embed = sinusoidal_embedding(t)

    logits = model([x_t_onehot, t_embed])  # (batch, n, n, 11)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        x_0_int, logits, from_logits=True
    )
    return tf.reduce_mean(loss)
```

This is arguably more principled for your setting since the output space
is genuinely discrete, and the loss directly measures tile prediction accuracy.

---

## 8. Exploring Architectures

### Things to vary and compare

| Hyperparameter         | What it controls                         | Try these          |
|------------------------|------------------------------------------|--------------------|
| `kernel_size`          | Receptive field / locality radius        | 3, 5, 7           |
| `filters` per layer    | Capacity of each layer                   | 32, 64, 128       |
| Number of down/up blocks | Depth of U-Net                        | 1, 2, 3           |
| `T` (timesteps)        | Granularity of diffusion                 | 50, 200, 1000     |
| Beta schedule           | How fast noise is added                  | linear, cosine    |
| Continuous vs discrete  | Type of diffusion process                | Gaussian vs D3PM  |

### Tracking experiments

```python
# Use TensorBoard for logging
log_dir = "logs/experiment_name"
summary_writer = tf.summary.create_file_writer(log_dir)

# Inside your training loop:
with summary_writer.as_default():
    tf.summary.scalar('loss', loss, step=global_step)

# Launch TensorBoard:
# $ tensorboard --logdir logs/
```

### Validation: checking mosaic validity

This is your secret weapon — you can *verify* outputs, which most generative
model papers can't do. After generation, check:

1. **Local compatibility**: Do adjacent tiles have matching connection points
   at shared edges? (This is a simple lookup on the 11 tile types.)
2. **Connectedness**: Is the resulting knot/link a single connected component?
3. **Knot identification**: What knot type was generated? (Use your
   KnotMosaics SageMath package.)

```python
def check_local_compatibility(mosaic):
    """
    mosaic: (n, n) integer array
    Returns fraction of adjacent pairs that are compatible.
    You'll fill in COMPAT_H and COMPAT_V from your tile definitions.
    """
    n = mosaic.shape[0]
    valid = 0
    total = 0

    for i in range(n):
        for j in range(n):
            # Check right neighbor
            if j + 1 < n:
                total += 1
                if (mosaic[i, j], mosaic[i, j+1]) in COMPAT_H:
                    valid += 1
            # Check bottom neighbor
            if i + 1 < n:
                total += 1
                if (mosaic[i, j], mosaic[i+1, j]) in COMPAT_V:
                    valid += 1

    return valid / total if total > 0 else 1.0
```

Track `check_local_compatibility` on generated mosaics during training
as a metric. Watching this go from ~random (1/11 ≈ 9%) toward 100% tells
you the model is learning the local rules.

---

## 9. Weight Transfer: 3×3 → 10×10

The key insight, and the core of the publishable result:

```python
# 1. Train on 3x3 mosaics
model_3x3 = make_unet(input_shape=(3, 3, 11))
train(model_3x3, dataset_3x3, params, epochs=200)

# 2. Build identical architecture for 10x10
#    (same layer structure, just different Input shape)
model_10x10 = make_unet(input_shape=(10, 10, 11))

# 3. Transfer weights — works because Conv2D weights are spatial-size-agnostic
model_10x10.set_weights(model_3x3.get_weights())

# 4. Generate and evaluate
for i in range(100):
    mosaic = generate(model_10x10, params, shape=(1, 10, 10, 11))[0]
    compat = check_local_compatibility(mosaic)
    knot_type = identify_knot(mosaic)  # your SageMath code
    print(f"Sample {i}: compatibility={compat:.2%}, knot={knot_type}")
```

**Important caveat with the U-Net**: MaxPooling2D(2) on a 3×3 input gives
a 1×1 bottleneck (after floor division). On a 10×10 input, the bottleneck
is 5×5. The spatial dimensions change, but the *weights* (filter tensors)
are the same shape and transfer directly. The model just applies them to
a larger canvas.

If you want to avoid any pooling issues with odd-sized inputs, you can
use a simpler "all-convolutional" architecture (no pooling at all):

```python
def make_simple_cnn(input_shape, time_embed_dim=32):
    """Purely convolutional — no pooling, no size constraints."""
    x_input = layers.Input(shape=input_shape)
    t_input = layers.Input(shape=(time_embed_dim,))

    t = layers.Reshape((1, 1, time_embed_dim))(t_input)
    t = tf.tile(t, [1, input_shape[0], input_shape[1], 1])
    x = layers.Concatenate()([x_input, t])

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    output = layers.Conv2D(NUM_TILE_TYPES, 1, padding='same')(x)

    return Model(inputs=[x_input, t_input], outputs=output)
```

This is the cleanest test of the locality/equivariance hypothesis: every
layer has kernel_size=3, so the effective receptive field after 4 layers
is 9×9. If this generates valid 10×10 mosaics, you've shown that a
receptive field of 9 suffices for knot mosaic generation.

---

## 10. Experiment Roadmap

A suggested sequence for getting results:

1. **Data generation**: Use KnotMosaics to enumerate valid 3×3 suitably
   connected mosaics. Export as integer arrays. How many are there?
   This determines whether you have enough training data.

2. **Sanity check**: Train on 3×3, generate 3×3. Does the model reproduce
   valid mosaics? What's the compatibility rate? This validates your
   pipeline before attempting generalization.

3. **The key experiment**: Transfer weights to 10×10. Generate many samples.
   Measure compatibility rate, connectedness, knot types. Compare against
   random baseline (uniform random tiles).

4. **Ablations**:
   - Kernel size 3 vs 5 vs 7: how does receptive field affect validity?
   - Continuous vs discrete diffusion: which produces more valid mosaics?
   - Number of layers: how much depth is needed?
   - One-hot vs scalar encoding: does it matter in practice?

5. **Analysis**: Distribution of generated knot types. Are there knots the
   model "discovers" that don't exist at 3×3? What's the maximum crossing
   number achieved? This is the headline result.

---

## Quick Reference: Key TF Functions

| Task                      | Function                                      |
|---------------------------|-----------------------------------------------|
| One-hot encode            | `tf.one_hot(tensor, depth=11)`                |
| Decode back to integers   | `tf.argmax(tensor, axis=-1)`                  |
| Convolution layer         | `layers.Conv2D(filters, kernel, padding)`     |
| Downsample                | `layers.MaxPooling2D(pool_size)`              |
| Upsample                  | `layers.UpSampling2D(size)`                   |
| Skip connection           | `layers.Concatenate(axis=-1)`                 |
| Build model               | `Model(inputs=[...], outputs=...)`            |
| Random noise              | `tf.random.normal(shape)`                     |
| Gradient computation      | `tf.GradientTape()`                           |
| Dataset from arrays       | `tf.data.Dataset.from_tensor_slices(arr)`     |
| Batching                  | `dataset.batch(batch_size)`                   |
| Save/load weights         | `model.save_weights()` / `model.load_weights()` |
| TensorBoard logging       | `tf.summary.scalar(name, value, step)`        |

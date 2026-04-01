## Autoencoders

### Motivation

So far we mostly considered **supervised learning**: every image comes with a label.
In reality, labeled data is scarce, while **unlabeled images are everywhere**.

Autoencoders give us a way to:

- Learn **compressed image representations** without labels.
- Detect **anomalies** (images that “don’t reconstruct well”).
- Perform **denoising** or **inpainting** by reconstructing a clean version.

The core idea: if a model can **reconstruct** an input image from a much smaller internal code, that code must capture meaningful structure of the image.

### Main idea

We will learn about **Autoencoders** and **Variational Autoencoders (VAEs)**.

Think of an autoencoder as a bottlenecked “copy machine”:

- **Encoder**: maps an image $x$ to a low-dimensional vector $z$ (the **latent code**).
- **Decoder**: maps $z$ back to an image $\hat{x}$.
- **Training objective**: make $\hat{x}$ as close as possible to $x$ (e.g. with mean squared error).

Intuition:

- If the bottleneck is **too wide**, the network can simply memorize pixels → poor representation.
- If the bottleneck is **too narrow**, the network must learn **compressed, structured features** that explain the image (edges, shapes, textures).

VAEs add a probabilistic twist:

- Instead of a single code, the encoder outputs **mean and variance** for a Gaussian over $z$.
- We **sample** $z$ from this distribution during training.
- This makes the latent space **smooth** and **generative**: nearby points correspond to similar images, and we can sample new images by sampling new $z$.

```{figure} https://lilianweng.github.io/posts/2018-08-12-vae/VQ-VAE.PNG
:width: 80%
:alt: Autoencoder-style encoder-decoder architecture

High-level encoder–decoder structure with a bottleneck latent representation $z$.
```




### Autoencoder architecture

We will focus on a **convolutional autoencoder** for images:

- **Encoder**
  - Stack of Conv → Nonlinearity → (optional) Pooling layers.
  - Spatial resolution shrinks, channel dimension often grows.
  - Final feature map is flattened to a latent vector $z$.

- **Decoder**
  - Starts from $z$, reshapes into a small feature map.
  - Uses ConvTranspose2d / upsampling layers to grow resolution back.
  - Final layer outputs a 1-channel (grayscale) or 3-channel (RGB) image.

Design choices:

- **Latent dimension** (size of $z$): smaller means stronger compression and more pressure to learn structure.
- **Depth**: more layers can capture more complex patterns but are harder to train.
- **Activation & normalization**: ReLU/LeakyReLU + BatchNorm/LayerNorm are common.

### Training

We train autoencoders with a **reconstruction loss**:

- For images with values in $[0, 1]$, a common choice is:
  - Mean squared error (MSE) between $x$ and $\hat{x}$, or
  - Per-pixel binary cross-entropy for binary-ish images.

For a **VAE**, the loss has two pieces:

- **Reconstruction loss**: encourages $\hat{x}$ to match $x$.
- **KL divergence**: encourages the learned latent distribution to be close to a simple prior (usually standard normal).

High-level training loop intuition:

1. Sample a batch of images.
2. Encode to latent codes $z$ (or mean/variance for VAE).
3. Decode to reconstructions $\hat{x}$.
4. Compute loss.
5. Backpropagate and update parameters.

### How to use an autoencoder?

Once trained, we can use an autoencoder in several ways:

- **Dimensionality reduction / feature extraction**  
  - Throw away the decoder and keep the encoder.  
  - Use the latent code $z$ as features for a simple downstream model (e.g. a small classifier).

- **Denoising**  
  - Train with **noisy inputs** and **clean targets**.
  - At test time, pass in noisy images and use the reconstruction as the denoised output.

- **Anomaly detection**  
  - Train on “normal” data only.
  - At test time, measure reconstruction error. Very high error suggests an out-of-distribution / anomalous sample.

- **Generation (with VAEs)**  
  - Sample $z$ from the prior and decode to generate **new** images.
  - Interpolate between two $z$ vectors to morph one image into another.


### The latent space of an autoencoder

The **latent space** of a trained autoencoder represents compressed, learned features of the input data. Each input image is mapped to a point (vector) in this lower-dimensional space.

#### Example: Latent space visualization with MNIST

Suppose we train an autoencoder on the MNIST handwritten digits dataset, using a 2D latent space ($z \in \mathbb{R}^2$). After training:

- Each digit image (e.g., '3', '7', '0') gets encoded to a unique 2D vector.
- If we plot these latent codes, we'll often see **clusters** emerge, with images of the same digit grouping together.
- This shows that the autoencoder has learned to summarize digit identity in just two numbers!

**Visualization:**
- Color each point by its true digit label.
- Clusters indicate that the latent variables have captured meaningful, semantically relevant structure.
- If we interpolate between points in latent space (move between two digit clusters), and decode, the output image smoothly morphs from one digit into another.

> **Key ideas:**  
> - The latent space captures "essence" instead of pixel-by-pixel information.  
> - Good autoencoders separate underlying factors (like digit identity) in the latent representation.

You can try this by training with `latent_dim=2`, collecting the latents for the test set, and plotting with matplotlib:

```python
import matplotlib.pyplot as plt

# Assume `autoencoder` is trained, `test_loader` yields (img, label)
latents = []
labels = []
with torch.no_grad():
    for x, y in test_loader:
        z = autoencoder.encode(x)
        latents.append(z.cpu().numpy())
        labels.append(y.cpu().numpy())
latents = np.concatenate(latents, axis=0)
labels = np.concatenate(labels, axis=0)
plt.figure(figsize=(6,6))
plt.scatter(latents[:,0], latents[:,1], c=labels, cmap='tab10', alpha=0.7, s=8)
plt.colorbar(label='Digit')
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.title("MNIST latent space (autoencoder, 2D)")
plt.show()
```

For higher-dimensional latent spaces, you can use PCA or t-SNE to project into 2D and visualize similar structure.


### Math formulation summary

#### Autoencoder: Objective and Equations

- **Encoder:** $ z = f_{\theta}(x) $, where $ x $ is the input and $ z $ is the latent code.
- **Decoder:** $ \hat{x} = g_{\phi}(z) $, where $ \hat{x} $ is the reconstructed input.

The autoencoder is trained to minimize the **reconstruction loss** between the input $ x $ and its reconstruction $ \hat{x} $. Commonly, mean squared error (MSE) is used:

$
\mathcal{L}_{\text{AE}}(\theta, \phi) = \mathbb{E}_{x \sim p_{\text{data}}(x)} \left[ \| x - g_{\phi}(f_{\theta}(x)) \|^2 \right]
$

---

#### Variational Autoencoder (VAE): Objective and Equations

- **Probabilistic encoder:** outputs mean and variance, parameterizing a Gaussian:
  $
  q_{\theta}(z\,|\,x) = \mathcal{N}(z\,|\,\mu_{\theta}(x),\,\sigma^2_{\theta}(x))
  $
- **Decoder:** generates a distribution over reconstructions:
  $
  p_{\phi}(x\,|\,z)
  $

The VAE maximizes a lower bound (ELBO) on the log likelihood, combining reconstruction and regularization:

$
\mathcal{L}_{\text{VAE}}(\theta, \phi) = \mathbb{E}_{x \sim p_{\text{data}}(x)} \left[
    \mathbb{E}_{z \sim q_{\theta}(z\,|\,x)} \left[ \log p_{\phi}(x\,|\,z) \right] 
    - D_{\mathrm{KL}}\big( q_{\theta}(z\,|\,x)\,\|\; p(z) \big)
\right]
$

Where:

- $ p(z) $ is a prior (usually standard normal $ \mathcal{N}(0, I) $)
- $ D_{\mathrm{KL}} $ is the Kullback-Leibler divergence (regularizes the latent space)

**Interpretation:**

- Encourage reconstructions to be accurate (**likelihood** term)
- Encourage the latent codes to be close to the prior (**KL** term), making the latent space continuous and structured

---






### Starter code

Below is a **minimal convolutional autoencoder** in PyTorch.
Focus on:

- How the encoder reduces spatial size.
- How the decoder mirrors the encoder.
- How the loss is defined and optimized.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()

        # Encoder: input (B, 1, 28, 28) -> latent (B, latent_dim)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
        )
        self.encoder_fc = nn.Linear(32 * 7 * 7, latent_dim)

        # Decoder: latent (B, latent_dim) -> reconstructed (B, 1, 28, 28)
        self.decoder_fc = nn.Linear(latent_dim, 32 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14 -> 28
            nn.Sigmoid(),  # outputs in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        z = self.encoder_fc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 32, 7, 7)
        x_hat = self.decoder_conv(h)
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


def train_autoencoder(model, dataloader, num_epochs=5, lr=1e-3, device=\"cuda\" if torch.cuda.is_available() else \"cpu\"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x, _ in dataloader:  # labels are ignored
            x = x.to(device)

            x_hat = model(x)
            loss = F.mse_loss(x_hat, x)  # reconstruction loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f\"Epoch {epoch+1}/{num_epochs} - recon loss: {avg_loss:.4f}\")


# In your notebook / script, you would create a DataLoader (e.g. MNIST),
# instantiate ConvAutoencoder(), and call train_autoencoder().
```


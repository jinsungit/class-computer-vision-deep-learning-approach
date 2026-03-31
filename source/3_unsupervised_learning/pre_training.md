## Pre-training for vision models

### Motivation

Large modern vision models (e.g. ViTs) are often trained on **billions of images**.
Labeling that many images is impossible, so we rely on **pre-training without labels** and then **fine-tuning** on specific tasks.

Idea: create an **artificial task** (a *pretext* task) that forces the model to understand images, even though no human labels are used.
Once the model is good at the pretext task, we reuse its weights for real downstream tasks (classification, detection, segmentation, etc.).

### Main idea

We will learn about **Masked Autoencoders (MAE)**.

MAE borrows the idea of **masked language modeling** from BERT and adapts it to images:

- Break the image into **patches** (e.g. 16×16).
- Randomly **mask** (hide) a large fraction of the patches (e.g. 75%).  
- The model sees only the visible patches and must **reconstruct the missing patches**.

Key intuition:

- To fill in missing patches, the model must build an **internal representation** of objects, textures, and scene layout.
- Because the masking is random, the model can’t just memorize; it must learn to **generalize** from context.

### Masked autoencoder architecture

At a high level, MAE uses a **Vision Transformer (ViT)** as an autoencoder:

- **Patch embedding**:
  - Split the image into fixed-size patches.
  - Flatten each patch and map it to an embedding vector (a “token”).

- **Masking**:
  - Randomly keep a subset of tokens (visible patches) and drop the others.
  - The encoder only processes visible tokens → **efficient**.

- **Encoder** (ViT encoder):
  - Runs self-attention on the visible tokens.
  - Produces contextualized embeddings that summarize the visible parts of the image.

- **Decoder**:
  - Re-inserts learned **mask tokens** in place of the hidden patches.
  - Runs a smaller transformer that attends over both visible and mask tokens.
  - Predicts pixel values (or patch features) for all patches.

```{figure} https://towardsdatascience.com/wp-content/uploads/2021/12/1l8zPV1sSDmEPbwHTDh5Rzw.png
:width: 80%
:alt: Masked autoencoder architecture

High-level MAE pipeline: mask patches, encode visible ones, then decode to reconstruct all patches.
```

### Training

Training MAE looks similar to training a classic autoencoder, but with masked inputs:

1. Take an image and convert it to patch embeddings.
2. Randomly select a subset of patches to **keep**; mark the rest as **masked**.
3. Run the encoder on the visible patches only.
4. Add mask tokens and run the decoder.
5. Compute reconstruction loss **only on the masked patches** (we don’t need to penalize visible ones).
6. Backpropagate and update parameters.

Important points:

- Masking ratio (e.g. 75%) is crucial: too low and the task is too easy; too high and the model has no context.
- Loss is typically MSE between predicted and true pixel values (or normalized/scaled version).
- The encoder is often kept relatively large; the decoder can be smaller (it’s only used during pre-training).

### How to use a masked autoencoder?

After pre-training, we **throw away the decoder** and keep the encoder as a **feature extractor**:

- **Fine-tuning for classification**:
  - Attach a classifier head (e.g. a linear layer) on top of the encoder’s CLS token or pooled features.
  - Fine-tune on labeled data (e.g. ImageNet, your own dataset).

- **Backbone for detection / segmentation**:
  - Use the encoder as the backbone in a detection/segmentation framework (e.g. Mask R-CNN, U-Net-style decoders).
  - Fine-tune the whole stack end-to-end.

- **Frozen features**:
  - Freeze the encoder and just train a small head on top if you have **very little labeled data**.

In all cases, the idea is: *pre-training on MAE gives you a strong starting point, so downstream training is more sample-efficient and reaches better accuracy*.

### Starter code

Below is a **toy example** in PyTorch that shows the structure of a masked patch autoencoder.
It does **not** implement a full ViT, but the flow mirrors MAE:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


class SimpleMAE(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=128, mask_ratio=0.75):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans=3, embed_dim=embed_dim)
        self.mask_ratio = mask_ratio

        # Tiny \"encoder\" and \"decoder\" just for illustration
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True),
            num_layers=2,
        )
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True),
            num_layers=1,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_predictor = nn.Linear(embed_dim, patch_size * patch_size * 3)

    def random_mask(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        num_keep = int(N * (1.0 - self.mask_ratio))

        rand = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(rand, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]
        ids_mask = ids_shuffle[:, num_keep:]

        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        return x_keep, ids_keep, ids_mask, N

    def forward(self, imgs):
        # imgs: (B, 3, H, W)
        patches = self.patch_embed(imgs)  # (B, N, D)
        x_keep, ids_keep, ids_mask, N = self.random_mask(patches)

        # Encode only visible patches
        enc = self.encoder(x_keep)  # (B, N_keep, D)

        # Reinsert mask tokens
        B, N_keep, D = enc.shape
        mask_tokens = self.mask_token.expand(B, N - N_keep, D)
        # Concatenate then unshuffle back to original order
        x_full = torch.cat([enc, mask_tokens], dim=1)

        # For simplicity, we skip the exact unshuffle logic here and just pass x_full
        dec = self.decoder(x_full)

        # Predict patch pixels
        patch_pred = self.patch_predictor(dec)  # (B, N, P*P*3)
        return patch_pred


def mae_pretrain_step(model, imgs):
    """
    imgs: (B, 3, H, W) in [0, 1]
    """
    patch_pred = model(imgs)
    # In practice, you would compute the loss only for masked patches
    # and compare to the ground-truth patch pixels.
    loss = patch_pred.mean() * 0  # placeholder for real loss
    loss.backward()
```

In practice, you would not write MAE from scratch; you would **load a pre-trained checkpoint**
and focus on fine-tuning it for your own dataset.
This toy example is just to connect the architecture diagram with code structure.


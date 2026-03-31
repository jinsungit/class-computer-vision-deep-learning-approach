## Vision Transformers (ViT)

### Summary
ViT is a direct adaptation of the Transformer architecture to images. To make it work on images, we need to:
- Split the image into patches.
- Flatten the patches into a sequence of tokens.
- Apply the Transformer architecture to the sequence of tokens.
- Reconstruct the image from the sequence of tokens.



### Main idea

At a high level, **ViT treats an image like a sentence**, and **small image patches like words**.

- Instead of reading one pixel at a time (like a CNN with tiny filters), ViT:
  - **Cuts the image into patches** (e.g., \(16 \times 16\) pixels).
  - **Flattens each patch** into a vector and projects it to an embedding (a token).
  - **Feeds the sequence of patch tokens into a Transformer encoder**, just like a sequence of word embeddings in NLP.
  - Uses **self-attention** so that each patch can look at all other patches and decide **which regions of the image are important to it**.

The key intuition:

- **CNNs** build locality and translation invariance through fixed convolution kernels and pooling.
- **ViTs** instead let **every patch attend to every other patch**, learning global relationships directly:
  - A patch on the left can immediately attend to one on the right.
  - Objects that are far apart in pixel space can influence each other in a single layer.

You can think of ViT as:

- **Input**: a "sentence" of image patches.
- **Model**: a Transformer that reasons over this sequence.
- **Output**: a prediction based on a special classification token, or on pooled information from all patches.

---

### Architecture

We will walk through the main building blocks from input image to output prediction.

```{figure} ./Vision_Transformer.gif
:width: 80%
:alt: ViT architecture

ViT architecture.
```

#### 1. Patch embedding

Imagine a \(224 \times 224\) RGB image and patch size \(16 \times 16\):

- Number of patches \(= (224 / 16) \times (224 / 16) = 14 \times 14 = 196\).
- Each patch has shape \(16 \times 16 \times 3\) (height, width, channels).
- We **flatten** each patch into a vector and pass it through a linear layer to get a **patch embedding**.

Visually:

```text
Image (224x224)
 └──> split into 16x16 patches
      [P1] [P2] [P3] ... [P196]
      └──> flatten + linear layer
           [E1] [E2] [E3] ... [E196]   (patch embeddings)
```

#### 2. Add a [CLS] token and positional encodings

We add:

- A learnable **[CLS] token** at the beginning of the sequence, whose final representation will summarize the whole image.
- **Positional encodings** so the model knows *where* each patch came from (top-left vs bottom-right, etc.).

```text
[CLS]  E1   E2   E3   ...   E196
  │     │    │    │          │
 +pos  +pos +pos +pos      +pos
  ↓
Sequence fed into Transformer encoder
```

#### 3. Transformer encoder blocks

We then stack several identical **Transformer encoder blocks**. Each block has two main sub-layers:

1. **Multi-Head Self-Attention (MHSA)**:
   - Every token (patch or [CLS]) looks at all tokens.
   - For each token, self-attention answers: **"Which other patches should I pay attention to, and by how much?"**
2. **Feed-Forward Network (MLP)** applied to each token independently:
   - A small neural network that further processes each token’s representation.

With residual connections and layer normalization, a block looks conceptually like:

```text
Input sequence
   │
   ├─> Multi-Head Self-Attention ─┐
   │                              │  (residual)
   └─────────────── add + norm ◄──┘
                 │
                 ├─> MLP (feed-forward) ─┐
                 │                       │  (residual)
                 └──────── add + norm ◄──┘
                 ↓
             Output sequence
```

Stacking many such blocks lets the model build **rich global features** about the image.

#### 4. Classification head

For image classification:

- Take the final representation of the **[CLS] token**.
- Pass it through a **linear layer (and maybe a softmax)** to get class probabilities.

```text
Final [CLS] representation ──> Linear layer ──> Class scores
```

---

### Example: ViT for image classification

Consider classifying images of **cats vs dogs**.

1. **Input image**: a 2D array of pixels.
2. **Patch embedding**:
   - The model cuts the image into patches and converts each patch into a token.
3. **Self-attention**:
   - A patch around a cat’s ear might attend strongly to patches around the eye and nose.
   - Background patches (e.g., sky, floor) might get low attention, since they are less informative.
4. **Deeper layers**:
   - Early layers learn lower-level visual patterns (edges, colors, textures).
   - Later layers learn higher-level concepts (ears, snouts, fur patterns, overall shape).
5. **[CLS] token**:
   - Aggregates information from all patches via attention.
   - Its final state should represent the whole image in a way useful for deciding "cat" or "dog".

Intuitively, self-attention gives ViT the ability to:

- **Look anywhere in the image at any time**.
- **Model long-range dependencies** (e.g., interaction between head and tail).
- **Adaptively focus** on the most relevant regions for a given task.

This global reasoning is especially powerful when:

- Objects are large, spread out, or occluded.
- Context from far-away regions is important.

---

### Starter code

Below is *pseudocode-style* PyTorch to illustrate the main steps. You do **not** need to understand every line; focus on how images become patch tokens and then go through a Transformer.

```python
import torch
import torch.nn as nn

class SimpleViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, emb_dim=256, depth=6, num_heads=8, num_classes=10):
        super().__init__()
        assert img_size % patch_size == 0
        num_patches = (img_size // patch_size) ** 2

        # 1. Patch embedding: unfold -> linear projection
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, emb_dim, kernel_size=patch_size, stride=patch_size)
        # Output shape: (B, emb_dim, H', W') with H'*W' = num_patches

        # 2. [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))

        # 3. Transformer encoder (using PyTorch's built-in module)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. Classification head on top of [CLS]
        self.mlp_head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        # Step 1: patch embedding
        x = self.proj(x)                 # (B, emb_dim, H', W')
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, emb_dim)

        # Step 2: prepend [CLS] and add positional encodings
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, emb_dim)
        x = torch.cat([cls_tokens, x], dim=1)           # (B, num_patches+1, emb_dim)
        x = x + self.pos_embed                          # add positional info

        # Step 3: Transformer encoder
        x = self.transformer(x)                         # (B, num_patches+1, emb_dim)

        # Step 4: classification head on [CLS]
        cls_rep = x[:, 0]                               # (B, emb_dim)
        logits = self.mlp_head(cls_rep)                 # (B, num_classes)
        return logits
```

This captures the **essence of ViT**:

- Turn images into a sequence of patch tokens.
- Use a Transformer encoder to let patches attend to each other.
- Use the [CLS] token (or pooled patches) for downstream tasks such as classification.

### Further reading

- [Vision Transformers](https://arxiv.org/abs/2010.11929)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Vision Transformers](https://arxiv.org/abs/2010.11929)
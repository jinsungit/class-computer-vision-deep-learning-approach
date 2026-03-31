## Self-supervised learning

### Motivation

Labeling images is expensive, but we can often create **supervision from the data itself**.
Self-supervised learning asks the model to solve tasks that require understanding images, **without any human labels**.

Examples of self-supervised tasks:

- Predict the **next frame** in a video.
- Predict the **relative position** of image patches.
- Predict whether two views come from the **same image**.

Once a model is good at such tasks, we can reuse it as a strong visual backbone.

### Main idea

We will learn about **SimCLR**, a widely used self-supervised contrastive learning method for images.

The core idea behind SimCLR (Simple Framework for Contrastive Learning of Visual Representations) is to leverage **contrastive learning**:  
The model is trained so that augmented views of the **same image** are close in the feature space, while views of **different images** are far apart.

How SimCLR works:

- For each image in a batch, generate **two different augmented views** via random transformations (crop, color jitter, blur, etc.).
- Pass each view through the same backbone network to extract features.
- Pass features through a small neural network called a **projection head**, which maps them to a space where contrastive loss is applied.
- Use a contrastive loss to maximize the agreement between features from the **same image** (positive pairs) and minimize it for features from **different images** (negative pairs).

This teaches the model to learn features that are **invariant** to data augmentations, capturing useful semantic information from images **without labels**.

### Self-supervised architecture

SimCLR architectural components:

- A shared **backbone** (e.g. ResNet or ViT) that extracts image features.
- A **projection head** (2-3 layer MLP) that maps features to a space for contrastive training.
- No separate teacher network (unlike DINO), just a single encoder+projection head trained end-to-end.

```{figure} https://pytorch.org/tutorials/_images/simclr.png
:width: 80%
:alt: SimCLR self-supervised learning pipeline

The SimCLR pipeline: two augmented views of each image are passed through the same encoder and projection head, then the representations are contrasted.
```

The illustration above shows how two random augmentations of each image feed into the same encoder, followed by the projection head, and the resulting vectors are compared using a contrastive loss.

### Training

SimCLR training usually follows this pattern:

1. Sample a batch of images.
2. For each image, create **two independent augmented views** (so with batch size $N$, you have $2N$ views).
3. Pass all $2N$ views through the backbone + projection head to get feature vectors.
4. For each positive pair (two views of the same image), the loss encourages them to be similar, while pushing other $2N-2$ features apart.  
   - This is done with the **NT-Xent** (normalized temperature-scaled cross-entropy) loss.
5. Backpropagate and update model parameters.

The key intuition:  
**learn visual features that are robust to augmentations and discriminative across different images, without any human labels**.

### How to use a self-supervised model?

Once trained, SimCLR-style models become strong **feature extractors**:

- **Linear probing**:
  - Freeze the backbone network.
  - Train a simple linear classifier on top with labeled data.
  - If the backbone learned good features, this classifier works well.
- **Fine-tuning**:
  - Initialize a model with SimCLR-pretrained weights.
  - Fine-tune all layers on your specific task.
  - Performance is usually better than training from scratch, especially with limited labels.

### Starter code

Below is a **toy SimCLR-style contrastive learning skeleton** in PyTorch.
This code demonstrates the core data flow, though a full SimCLR implementation would require larger batch sizes and richer augmentations.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyBackbone(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(64 * 8 * 8, feature_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class SimCLR(nn.Module):
    def __init__(self, feature_dim=128, projection_dim=64):
        super().__init__()
        self.backbone = TinyBackbone(feature_dim=feature_dim)
        self.projection_head = ProjectionHead(feature_dim, projection_dim)

    def forward(self, x):
        features = self.backbone(x)
        proj = self.projection_head(features)
        proj = F.normalize(proj, dim=-1)
        return proj

def nt_xent_loss(z, temperature=0.5):
    """
    Normalized Temperature-Scaled Cross Entropy Loss for SimCLR.
    z: (2N, D) - concatenated projections for both augmented views in the batch.
    Returns scalar loss.
    """
    N = z.shape[0] // 2
    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.T) / temperature     # (2N, 2N)
    labels = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0).to(z.device)
    mask = ~torch.eye(2*N, dtype=bool, device=z.device)
    sim = sim.masked_select(mask).view(2*N, -1)
    loss = F.cross_entropy(sim, labels)
    return loss

def simclr_step(model, batch_view1, batch_view2):
    """
    One training step for SimCLR-style contrastive learning.

    batch_view1, batch_view2: two differently augmented views of the same images,
    both of shape (B, 3, H, W).
    """
    # Concatenate two augmented views along the batch dimension
    x = torch.cat([batch_view1, batch_view2], dim=0)  # (2B, 3, H, W)
    z = model(x)  # (2B, D)
    loss = nt_xent_loss(z)
    loss.backward()
    return loss.item()
```

In actual research code, you would use a larger backbone (ResNet, ViT), stronger augmentations, much bigger batch sizes (since more negatives help), and train for many epochs.

**Bottom line:** by matching representations of augmented views from the same image and contrasting them with others, SimCLR trains powerful, label-efficient visual features.


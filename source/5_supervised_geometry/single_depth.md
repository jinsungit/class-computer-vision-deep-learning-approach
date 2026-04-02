## Single-view depth estimation

### Motivation

From **one** photograph you only observe a **single projection** of the 3D scene. Classical stereo and structure-from-motion recover depth by triangulating **corresponding points across views**. With a single image, that correspondence signal is missing, so depth is **underconstrained**: many different 3D scenes can produce the same image.

Nevertheless, humans infer **rough layout** (near vs far, ground plane, object boundaries) using **monocular cues**: perspective, texture gradients, relative sizes of familiar objects, occlusion, shading, and defocus. Supervised depth networks learn a similar prior from data: map appearance patterns to plausible depth maps.

Why it matters in practice:

- **Robotics and navigation** need obstacle distances from a monocular camera.
- **AR and graphics** need depth for occlusion and relighting.
- **Photo and video editing** uses depth for synthetic defocus and 3D-aware effects.
- It is a clean **pixel-level supervised** task: dense targets align naturally with dense network outputs.

Intuition: the network does not “invert optics” in closed form; it **predicts** a depth field that is **consistent with training examples** and generalizes to new scenes when cues resemble what was seen during training.

```{figure} https://upload.wikimedia.org/wikipedia/commons/a/ad/Ponzo_illusion.svg
:width: 42%
:alt: Ponzo illusion: parallel lines on receding rails

Monocular **depth and layout cues** can be powerful even when metric distance is ambiguous. The Ponzo illusion: the two horizontal segments have the same length, but perspective suggests different “distances,” so the brain interprets sizes differently—illustrating how **context** shapes perceived 3D structure. Depth networks exploit similar statistical regularities from data. *Image: PolBr, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en), Wikimedia Commons.*
```

---

### Main idea

**Task.** Given an RGB image $I$, predict a **dense depth map** $D$ of the same spatial resolution (or upsampled to full resolution), where each pixel stores distance along the viewing ray—usually **$Z$ in the camera frame** (positive in front of the camera), or an equivalent such as **inverse depth** or **disparity**.

**Disparity vs depth.** For a pinhole with baseline $b$ and focal length $f$, stereo disparity $d$ (in pixels) relates to depth by $Z \propto 1/d$. Many networks predict **inverse depth** $\eta = 1/Z$ because it varies more slowly across typical indoor/outdoor scenes and behaves better under gradient-based optimization.

**Metric vs relative depth.**

| Setting | What is predicted | Typical supervision |
|--------|-------------------|---------------------|
| **Metric depth** | Meters (or same units as sensor) | LiDAR, RGB-D cameras, synthetic renders with known scale |
| **Affine-invariant / relative** | Correct up to unknown scale (and sometimes shift) | In-the-wild video, stereo without absolute calibration |
| **Scale-aligned at test time** | Relative map scaled to match sparse GT | Common on benchmarks when only relative training is available |

**Mistake to avoid:** comparing two methods on **raw** predicted depth when one outputs **metric** depth and another is **scale-ambiguous**. For fair comparison, apply **median scaling** (or least-squares scale alignment) on the predicted map using ground-truth valid pixels, then compute metrics—unless the task strictly requires absolute meters everywhere.

```mermaid
flowchart LR
  I[RGB image] --> F[Encoder]
  F --> D[Decoder / head]
  D --> Z[Depth or inv-depth per pixel]
```

---

### Model

The dominant pattern is an **encoder–decoder** with **skip connections** so that fine boundaries in the input are preserved in the output:

- **Encoder**: ImageNet-pretrained CNN (ResNet, EfficientNet) or Vision Transformer blocks extract multi-scale features.
- **Decoder**: Progressive upsampling (bilinear + convolutions, or transposed convolutions) fuses low-resolution semantic context with high-resolution spatial detail—conceptually similar to U-Net and semantic segmentation heads.

**Multi-scale prediction** (e.g., Eigen *et al.*): predict depth at several resolutions and combine, which stabilizes training for very deep scenes where global context matters.

**Representative families** (names for reading, not an exhaustive survey):

- **Convolutional encoder–decoders** with skip connections for indoor (NYU) and outdoor (KITTI) benchmarks.
- **Transformer-based depth** (e.g., DPT-style): patch tokens and fused feature pyramids for strong zero-shot transfer when pretrained on large mixed datasets.
- **Foundation depth models** (e.g., MiDaS lineage): train on heterogeneous depth sources to learn a **general monocular prior**, often evaluated after **affine alignment** to each dataset’s scale.

For course projects, a **ResNet backbone + lightweight decoder** with supervised L1 or scale-invariant loss on NYU Depth v2 or a KITTI depth split is a standard, reproducible baseline.

```{figure} https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png
:width: 95%
:alt: Contracting CNN path with convolutions and pooling

**Encoder intuition:** spatial resolution shrinks while the channel dimension grows, building a **hierarchical** representation from local edges to broader context. A **depth decoder** adds a symmetric expanding path (upsampling + skips, as in U-Net) to recover a **dense** depth map at full resolution. *Image: Aphex34, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en), Wikimedia Commons.*
```

---

### Training

**Targets.** Ground-truth depth $D^\star$ may be **sparse** (LiDAR scans) or **dense** (structured light, Kinect, rendered). Training masks out **invalid** pixels (no return, out of range, sky labels if applicable).

**Common losses** (per valid pixel, possibly averaged):

| Loss | Formula (conceptually) | Notes |
|------|-------------------------|--------|
| **L1 / L2** on $Z$ or $\log Z$ | $\|D - D^\star\|$ | Simple; sensitive to outliers in depth |
| **BerHu** | Huber on large errors | Behaves like L2 near the minimum, robust far from it |
| **Scale-invariant (Eigen)** | $\frac{1}{n}\sum_i (\log D_i - \log D^\star_i + \alpha)^2$ with $\alpha$ chosen to minimize the sum over the batch/patch | Encourages **relative** structure; helps when global scale drifts |

Many pipelines also add a **gradient matching** term on $\nabla D$ vs $\nabla D^\star$ to sharpen edges, or an **edge-aware smoothness** term on $D$ weighted by image gradients (encourage smooth depth except at color edges).

**Augmentation:** color jitter, horizontal flips (if depth is defined consistently), random crops, and **random scale** of depth together with intrinsics-aware scaling when simulating focal length changes—keep augmentation consistent with your label definition.

**Optimization:** AdamW or Adam, learning rate warmup + cosine decay is common; longer training on diverse data often matters as much as architecture tweaks.

---

### Evaluation

Standard **depth completion** metrics (NYU / KITTI style) compare predicted $D$ and ground truth $D^\star$ on **valid** pixels only. Let $e_i = D_i - D^\star_i$ and $r_i = D_i / D^\star_i$ (with care when $D^\star_i$ is small).

| Metric | Meaning |
|--------|--------|
| **Abs Rel** | $\frac{1}{|M|}\sum_{i\in M} |e_i| / D^\star_i$ — relative error, scale-free |
| **Sq Rel** | $\frac{1}{|M|}\sum_{i\in M} e_i^2 / D^\star_i$ |
| **RMSE** | $\sqrt{\frac{1}{|M|}\sum_{i\in M} e_i^2}$ — in meters if $D$ is metric |
| **RMSE log** | $\sqrt{\frac{1}{|M|}\sum_{i\in M} (\log D_i - \log D^\star_i)^2}$ |
| **$\delta_k$** | Fraction of pixels with $\max(r_i, 1/r_i) < 1.25^k$ for $k \in \{1,2,3\}$ — “good” pixels under ratio threshold |

Higher $\delta_1$ and lower Abs Rel / RMSE are better. Report **which mask** you use (e.g., KITTI crop, max depth cap) so numbers are comparable across papers.

**Qualitative checks:** look at **edges** (depth should align with object boundaries), **thin structures** (poles, furniture legs), and **textureless regions** (walls, sky) where ambiguity is worst.

---

### Math formulation summary

Let $\theta$ denote network parameters. A generic supervised objective is

$$
\mathcal{L}(\theta) = \frac{1}{|M|} \sum_{(u,v)\in M} \ell\big(D_\theta(u,v), D^\star(u,v)\big) + \lambda \, \mathcal{R}(D_\theta, I),
$$

where $M$ is the set of valid pixels, $\ell$ is L1, BerHu, or scale-invariant log loss, and $\mathcal{R}$ is an optional smoothness or gradient term. If predictions are **affine-invariant**, training may include a per-image or per-batch **scale (and bias) alignment** so that only **relative** structure is penalized.

---

### Starter code

Minimal PyTorch-style sketch: encoder–decoder returns a depth map; L1 on valid pixels only.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):
    """Example: lightweight decoder on top of a pretrained backbone."""

    def __init__(self, backbone_out_ch: int = 2048):
        super().__init__()
        self.reduce = nn.Conv2d(backbone_out_ch, 256, kernel_size=1)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.reduce(feat)
        return self.up(x).squeeze(1)  # (B, H', W')


def depth_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """pred, target: (B,H,W); valid: (B,H,W) bool or 0/1."""
    v = valid.float()
    return (v * (pred - target).abs()).sum() / v.sum().clamp_min(1.0)
```

Suggested student exercises:

1. Plug in `torchvision.models.resnet50` and build skip connections from intermediate layers (true U-Net-style) instead of a single low-res tensor.
2. Implement **scale-invariant log loss** on a random crop and compare to plain L1 on NYU Depth v2.
3. Log **Abs Rel** and **$\delta_1$** each epoch; visualize predicted vs GT depth maps side by side.
4. Augment with horizontal flip: verify depth tensor flips consistently with the image.

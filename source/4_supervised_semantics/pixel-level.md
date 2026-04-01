## Pixel-level label (segmentation)

### Motivation

Image-level labels tell us *what* is in an image, but not *where*.
For many real applications, we need dense, spatially precise understanding:

- **Medical imaging**: outline a lung nodule or the liver boundary so a radiologist can measure volume and track change over time.
- **Autonomous driving**: label road surface, lane markings, pedestrians, and vehicles so planning knows what is drivable vs obstacle.
- **Robotics**: segment “graspable object” vs “table” so motion planning avoids collisions.
- **Photo editing**: produce a tight mask around hair or a product so the background can be replaced without halos.

This is why segmentation is central in modern vision: it converts an image into a **pixel-wise semantic map**—a label at every location, not just one label for the whole frame.

### Main idea

In segmentation, the model predicts a class for **every pixel**:

- Input: image $x \in \mathbb{R}^{H \times W \times 3}$
- Output: a dense label map $y \in \{1, \ldots, C\}^{H \times W}$ (or a binary foreground mask when $C=2$)

Intuition-first view:

1. The encoder extracts increasingly abstract features (textures $\rightarrow$ parts $\rightarrow$ objects).
2. Spatial resolution is reduced in the encoder, so each location “sees” a large receptive field (context).
3. A decoder upsamples features back to full resolution so each pixel can be classified locally.
4. The final layer outputs $C$ logits per pixel; softmax over classes gives $p(\text{class} \mid \text{pixel})$.

Two common tasks:

- **Semantic segmentation**: all pixels labeled by *category*—every car shares the class “car,” and instances are not distinguished.
- **Instance segmentation**: pixels are grouped by *object instance*—car A and car B have different IDs even if both are “car” at the semantic level.

Concrete contrast: in a street photo, semantic segmentation might paint every person the same color; instance segmentation would give each pedestrian a distinct mask so a counting or tracking system can tell them apart.

```{figure} https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.PNG
:width: 80%
:alt: U-Net architecture for segmentation

U-Net intuition: downsample to capture context, then upsample with skip connections to recover spatial detail.
```

### Architecture

We will focus on **U-Net** for segmentation, and briefly connect it to modern foundation models.

#### U-Net

U-Net is popular because it balances global context and local detail:

- **Encoder path (contracting)**:
  - Convolution blocks + downsampling.
  - Builds high-level semantic features.
- **Decoder path (expanding)**:
  - Upsampling + convolution blocks.
  - Recovers fine spatial resolution.
- **Skip connections**:
  - Copy features from encoder to decoder at matching scales.
  - Preserve boundaries and small structures that might disappear during downsampling.

Why students like U-Net:

- Easy to understand and implement.
- Works well on small/medium datasets.
- Strong baseline for both natural and medical images.

#### Segment Anything (SAM) mindset

Modern segmentation systems like SAM shift from “train one model per dataset” to “promptable segmentation”:

- Use large-scale pretraining.
- Accept prompts (points, boxes, masks, text in some variants).
- Generalize to unseen images with minimal task-specific tuning.

You can think of U-Net as a task-specific workhorse and SAM-like models as broad, reusable segmentation foundations.

### Training

A practical segmentation training pipeline:

1. **Data and masks**
   - Verify mask quality and class definitions.
   - Align image/mask transforms exactly (same crop/flip); a one-pixel shift between image and mask ruins learning.
2. **Preprocessing**
   - Resize or crop to a fixed training size.
   - Normalize images; keep masks as integer class IDs.
3. **Loss**
   - Cross-entropy for multi-class segmentation.
   - Dice loss (or CE + Dice) for class imbalance and boundary sensitivity.
4. **Optimization**
   - AdamW/SGD, LR schedule, weight decay.
5. **Validation**
   - Monitor IoU/mIoU and Dice, not only pixel accuracy (see **Evaluation**).

What students should watch:

- **Class imbalance** (tiny objects disappear): use Dice/Focal loss, class weighting, or oversampling patches that contain rare classes.
- **Boundary quality**: check qualitative overlays (colored mask on image), not only scalar metrics.
- **Overfitting**: strong train metrics but poor val masks $\rightarrow$ more augmentation or regularization.

Common augmentations:

- Random crop, flip, mild color jitter.
- Scale jitter and elastic deformation (especially for medical data).

### Evaluation

Segmentation is evaluated on **regions**, not single labels. A model can get 95% pixel accuracy yet fail on a thin class (e.g., poles) or along boundaries—so we report overlap-based metrics and look at images.

#### Intersection over Union (IoU)

For one class, treat prediction and ground truth as binary masks. **Intersection** is pixels where both are foreground; **union** is pixels where either is foreground.

$$
\text{IoU} = \frac{|\text{prediction} \cap \text{ground truth}|}{|\text{prediction} \cup \text{ground truth}|}
= \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}.
$$

IoU is 1 when the masks match exactly, and 0 when they do not overlap. It penalizes both missed regions and spurious blobs.

```{figure} https://upload.wikimedia.org/wikipedia/commons/c/c7/Intersection_over_Union_-_visual_equation.PNG
:width: 72%
:alt: Intersection over Union equals intersection area divided by union area

IoU is the ratio of overlap to union (identical to the Jaccard index for binary masks). The same definition applies per class in multi-class segmentation by treating each class as a binary mask versus the rest.
```

```{figure} https://upload.wikimedia.org/wikipedia/commons/e/e6/Intersection_over_Union_-_poor%2C_good_and_excellent_score.PNG
:width: 85%
:alt: Three bounding boxes showing poor, good, and excellent IoU scores

Higher IoU means tighter agreement between prediction and ground truth. Segmentation masks use the same IoU idea on pixel regions instead of boxes.
```

#### Mean IoU (mIoU)

For $C$ foreground classes (often **excluding** a void/ignore class), compute IoU per class, then average:

$$
\text{mIoU} = \frac{1}{C}\sum_{c=1}^{C} \text{IoU}_c.
$$

**Why mIoU matters**: pixel accuracy is dominated by large “easy” classes (e.g., sky, road). mIoU forces every class—including small or rare ones—to contribute equally to the score.

#### Dice coefficient (and Dice loss)

For binary masks, the **Dice** score measures overlap in a way that is sensitive to thin structures:

$$
\text{Dice} = \frac{2|\text{prediction} \cap \text{ground truth}|}{|\text{prediction}| + |\text{ground truth}|}
= \frac{2\,\mathrm{TP}}{2\,\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}.
$$

Training sometimes minimizes **Dice loss** $= 1 - \text{Dice}$ (with a small $\epsilon$ in denominators for stability). Dice is closely related to the **F1** score for the positive class and is a common complement to cross-entropy when foreground pixels are scarce.

#### Pixel accuracy (use with caution)

$$
\text{Pixel accuracy} = \frac{\text{\# pixels correctly classified}}{\text{\# pixels}}.
$$

If 90% of pixels are “background,” a trivial predictor can score 90% without understanding objects. Always pair pixel accuracy with **per-class** metrics (IoU, Dice) or **balanced** measures.

#### What to look at in practice

| Metric | What it rewards | Typical pitfall |
|--------|-----------------|-----------------|
| Pixel accuracy | Getting many pixels right | Ignores rare classes and boundary errors |
| mIoU | Fair treatment across classes | Classes with no predicted pixels can get IoU $=0$ and drag the mean |
| Dice (per class) | Overlap for each binary mask | Still need to aggregate across classes for a single number |
| Qualitative overlays | Boundaries, small objects, confusion pairs | Subjective; use alongside numbers |

**Concrete intuition**: imagine segmenting “road” vs “sidewalk” in a city scene. A model that bleeds road pixels onto the sidewalk might still have high pixel accuracy if road dominates the image; IoU on the “sidewalk” class would drop sharply, flagging the failure.

### Math formulation summary

For $C$-class segmentation, the model outputs per-pixel logits $s_{ij} \in \mathbb{R}^{C}$ at each spatial location $(i, j)$ (in practice a tensor of shape $(B, C, H, W)$).

**Per-pixel probabilities** (softmax across classes at each pixel):

$$
p_{ij}(c) = \frac{\exp(s_{ij,c})}{\sum_{k=1}^{C}\exp(s_{ij,k})}.
$$

**Cross-entropy loss** (mean over pixels; $y_{ij}$ is the ground-truth class index at $(i,j)$):

$$
\mathcal{L}_{\text{CE}} =
-\frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}\log p_{ij}(y_{ij}).
$$

**Dice** (binary case; $\hat{m}, m \in \{0,1\}^{H \times W}$ are predicted and ground-truth masks, sums count foreground pixels):

$$
\text{Dice} =
\frac{2\sum_{i,j} \hat{m}_{ij} m_{ij} + \epsilon}{\sum_{i,j} \hat{m}_{ij} + \sum_{i,j} m_{ij} + \epsilon},
\qquad
\mathcal{L}_{\text{Dice}} = 1 - \text{Dice}.
$$

Interpretation:

- **CE** pushes each pixel toward the correct class label; it is the standard multi-class segmentation objective.
- **Dice** emphasizes **regional overlap** and is often used when foreground pixels are rare or when boundary alignment matters.

### Starter code

Below is a compact PyTorch U-Net-style scaffold for teaching.
It is intentionally small so students can read and modify it.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)  # concat skip: 64 + 64
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)   # concat skip: 32 + 32

        self.head = nn.Conv2d(32, num_classes, kernel_size=1)  # per-pixel logits

    def forward(self, x):
        e1 = self.enc1(x)          # (B, 32, H, W)
        e2 = self.enc2(self.pool1(e1))  # (B, 64, H/2, W/2)
        b = self.bottleneck(self.pool2(e2))  # (B, 128, H/4, W/4)

        d2 = self.up2(b)           # (B, 64, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)          # (B, 32, H, W)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.head(d1)     # (B, C, H, W)
        return logits


def train_step(model, images, masks, optimizer):
    """
    images: (B, 3, H, W)
    masks:  (B, H, W) with class indices in [0, C-1]
    """
    model.train()
    logits = model(images)
    loss = F.cross_entropy(logits, masks.long())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def predict_mask(model, images):
    model.eval()
    logits = model(images)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)  # (B, H, W)
    return probs, pred
```

Suggested student exercises:

1. Add Dice loss and compare with CE-only training.
2. Visualize predicted masks overlaid on input images every epoch.
3. Replace `TinyUNet` with a pretrained encoder U-Net and compare mIoU.
4. Try SAM on the same dataset and compare annotation effort vs performance.

### Advanced reading

- [Segment Anything](https://github.com/facebookresearch/segment-anything)

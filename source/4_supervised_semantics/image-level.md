## Image-level label (classification and tagging)

### Motivation

Image-level supervision is the most common entry point to modern vision:

- **Classification**: predict one label per image (e.g., cat, dog, car).
- **Tagging / multi-label classification**: predict multiple labels per image (e.g., beach, sunset, person, boat).

At a glance: single-label problems use a **partition** of images into classes; multi-label problems use **overlapping** attributes or objects. That distinction drives both how you encode labels and which loss you use (see the table under **Main idea**).

Why this matters:

- It is one of the most scalable supervised setups.
- Features learned for image-level labels transfer well to detection, segmentation, retrieval, and vision-language systems.
- Many practical products rely on this setup: content filtering, product categorization, medical triage, and quality inspection.

Intuition:
we ask the model to summarize the whole image into a semantic representation, then map that representation to labels.

### Main idea

For image-level tasks, the network learns a function:

- Input: image $x$
- Output: either a **single** categorical distribution over mutually exclusive classes, or **independent** per-class scores for tags that can co-occur.

Keep two things separate: **head structure** (how logits become probabilities) and **loss** (how predictions are scored against targets). The backbone is often the same; what changes is the last layer’s semantics and the training objective.

A useful mental model for the representation:

1. Early layers detect simple visual patterns (edges, corners, textures).
2. Middle layers combine patterns into parts and motifs.
3. Deeper layers encode object/category-level semantics.
4. A final prediction head turns those semantics into label logits (then softmax or sigmoid, depending on the task).

#### Single-label vs multi-label: structure and loss

| | **Single-label classification** | **Multi-label tagging** |
|---|--------------------------------|---------------------------|
| **When to use** | Exactly one label is correct per image (taxonomy is a partition). | Any subset of labels can be correct; labels can appear together. |
| **Target you store** | One class index per image, or a one-hot vector of length $C$. | A binary vector of length $C$ (1 = present, 0 = absent). |
| **Head structure** | One vector of $C$ logits; **softmax** turns them into a distribution that sums to 1. | One logit **per class**; each class gets its own **sigmoid** (no competition across classes). |
| **Loss** | **Cross-entropy** on the true class (equivalently, softmax + NLL). | **Binary cross-entropy** per class (average or sum over $C$); each dimension is its own yes/no decision. |
| **Prediction rule** | `argmax` over classes (or top-$k$ for evaluation). | Threshold sigmoid probabilities (e.g., $> 0.5$) per tag; different tags can fire together. |

Concrete intuitions:

- **Single-label**: “This product belongs to exactly one aisle: dairy, produce, or bakery.” A photo of a yogurt cup should not receive probability mass on both dairy and bakery; softmax forces the model to allocate evidence across competing options.
- **Multi-label**: “Which of these attributes appear: {glasses, smile, outdoor, night}?” A person can wear glasses and smile outdoors at dusk—several tags are 1 at once, so each tag needs its own probability, not a single winner.

Same backbone, different head semantics: in code, both setups often use `Linear(in_features, C)`; the difference is **softmax + CE** vs **sigmoid + BCE**, and whether `targets` are class indices or a `(B, C)` binary matrix.

```{figure} https://learnopencv.com/wp-content/uploads/2023/01/Convolutional-Neural-Networks.png
:width: 80%
:alt: CNN feature hierarchy

Feature hierarchy intuition: low-level patterns are composed into high-level semantics used for image-level prediction.
```

### Architecture

Two dominant backbones are still central in practice: **ResNet** and **Vision Transformer (ViT)**.
Both can produce strong image-level classifiers; the best choice often depends on data scale, compute budget, and pretraining availability.

#### ResNet

ResNet introduced residual connections, which made deep networks easy to optimize.

High-level structure:

- Convolutional stem + stack of residual blocks.
- Each residual block learns a residual function added to its input.
- Global average pooling turns spatial feature maps into a single feature vector.
- Linear head maps features to class logits.

Why it works well:

- Residual connections stabilize gradient flow.
- Convolutions naturally capture local spatial structure.
- Strong performance with moderate compute and robust transfer behavior.

Typical use in class projects:

- Start from ImageNet-pretrained `resnet18` or `resnet50`.
- Replace the last classification layer with your dataset's class count.
- Fine-tune with moderate augmentation.

#### Vision Transformer

ViT treats an image as a sequence of patch tokens.

High-level structure:

- Split image into patches (e.g., 16x16).
- Project each patch to an embedding vector.
- Add positional embeddings and pass through transformer encoder blocks.
- Use a class token (or pooled token features) for final prediction.

Why it is important:

- Scales very well with data/model size.
- Compatible with many modern pretraining pipelines (MAE, DINO, CLIP-style objectives).
- Strong transfer when initialized from large-scale pretraining.

Practical tradeoff:

- With small datasets and no pretraining, CNNs can be easier.
- With strong pretrained checkpoints, ViTs are often excellent and flexible.

### Training

A practical, intuition-first training pipeline:

1. **Data curation**
   - Verify labels, remove obvious noise, define class taxonomy.
2. **Preprocessing**
   - Resize/crop images, normalize with model-specific mean/std.
3. **Augmentation**
   - Random crop, flip, color jitter; optionally mixup/cutmix.
4. **Optimization**
   - AdamW or SGD, appropriate learning rate schedule, weight decay.
5. **Validation**
   - Track accuracy/F1 and calibration trends; save best checkpoint.

What students should watch:

- **Underfitting**: training and validation both low -> model/data pipeline too weak.
- **Overfitting**: training high, validation low -> add augmentation/regularization, reduce head capacity, or get more data.
- **Class imbalance**: use weighted loss, resampling, or macro metrics.

Metrics:

- **Single-label**: top-1 accuracy, top-k accuracy (does the highest-probability class match the one true label?).
- **Multi-label**: precision/recall/F1 (micro and macro), mAP (do predicted tag sets align with the ground-truth binary vector?—accuracy alone is often misleading because many label combinations are valid).

#### Finetuning

Fine-tuning is usually the most effective default for class projects.

Recommended staged strategy:

1. **Linear probing (warm start)**
   - Freeze backbone, train only classifier head for a few epochs.
2. **Partial unfreezing**
   - Unfreeze last block(s), continue with smaller learning rate.
3. **Full fine-tuning**
   - Unfreeze all layers with discriminative learning rates
     (smaller for backbone, larger for new head).

Why this helps:

- Head quickly adapts to your label space.
- Backbone adapts gradually, avoiding catastrophic forgetting.

Checklist for robust fine-tuning:

- Use pretrained checkpoints whenever possible.
- Keep an eye on validation curves after each unfreeze stage.
- Save best model by validation metric (not just final epoch).

### Math formulation summary

Let $x$ be an image. Write $s \in \mathbb{R}^C$ for the logits from the final linear layer (same tensor shape in both setups; the difference is how $s$ is turned into probabilities and matched to targets).

#### Single-label classification

- **Structure**: map $s$ to a categorical distribution with softmax over the $C$ classes.
- **Targets**: $y_{\text{true}} \in \{1,\ldots,C\}$ (or a one-hot vector).
- Probabilities:
  $$
  p(y=c\mid x)=\frac{\exp(s_c)}{\sum_{j=1}^{C}\exp(s_j)}.
  $$
- **Loss (cross-entropy)**:
  $$
  \mathcal{L}_{\text{CE}} = -\log p(y_{\text{true}}\mid x).
  $$

#### Multi-label tagging

- **Structure**: map each component $s_c$ to its own probability with **sigmoid** (no normalization across $c$).
- **Targets**: $y \in \{0,1\}^C$ with $y_c=1$ if tag $c$ is present.
- Per-class probabilities:
  $$
  p_c = \sigma(s_c).
  $$
- **Loss (binary cross-entropy)**, typically averaged over classes:
  $$
  \mathcal{L}_{\text{BCE}} = -\frac{1}{C}\sum_{c=1}^{C}
  \big[y_c\log p_c + (1-y_c)\log(1-p_c)\big].
  $$

Interpretation:

- **Softmax + CE** encodes mutual exclusivity: increasing the score for one class lowers the others’ implied probability mass.
- **Sigmoid + BCE** encodes independent presence/absence: “sunset” and “ocean” can both be high without forcing “beach” down.

**Mistake to avoid**: using softmax + CE when each image can have multiple true classes. The loss assumes a single winner; multi-hot targets do not match that assumption. Use sigmoid + BCE (or a reduction that treats each class as a binary problem) instead.

### Starter code

Below is a teaching-friendly PyTorch scaffold with:

- A pretrained ResNet backbone (same `Linear` head shape for both modes).
- Switchable **targets + loss**: integer class ids + `cross_entropy` vs binary `(B, C)` + `binary_cross_entropy_with_logits`.
- A minimal training step students can extend.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImageLevelClassifier(nn.Module):
    def __init__(self, num_classes: int, multi_label: bool = False):
        super().__init__()
        self.multi_label = multi_label

        # Pretrained backbone for fast convergence and better generalization
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # logits


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, multi_label: bool) -> torch.Tensor:
    if multi_label:
        # targets shape: (B, C) with 0/1 values
        return F.binary_cross_entropy_with_logits(logits, targets.float())
    # single-label targets shape: (B,) with class ids
    return F.cross_entropy(logits, targets.long())


def train_one_epoch(model, dataloader, optimizer, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.train()
    model.to(device)

    total_loss = 0.0
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = compute_loss(logits, targets, model.multi_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def predict(model, images, threshold=0.5):
    model.eval()
    logits = model(images)

    if model.multi_label:
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).int()
        return probs, preds
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    return probs, preds
```

Suggested student exercises:

1. Replace `resnet18` with `resnet50` and compare validation accuracy/time.
2. Add data augmentation and quantify its effect.
3. Convert a single-label dataset into a multi-label toy setting and compare CE vs BCE behavior.
4. Add a confusion matrix and inspect which classes the model confuses most.

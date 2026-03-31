## Object-level label (object detection)

### Motivation

Image-level labels answer *what is present*; pixel labels answer *what class each pixel is*.
Many applications need something in between: **where each distinct object is**, together with its category, without a full mask.

Object detection provides **category + location**, usually as **axis-aligned bounding boxes** (and often a confidence score per box).

Why this matters:

- **Autonomous driving**: localize vehicles, pedestrians, and signs in time for planning; a single “car” label for the whole frame is not enough to brake for the cyclist in the adjacent lane.
- **Retail / robotics**: find product instances on a shelf or a tool on a bench so a gripper can aim at a box, not the entire scene.
- **Video analytics**: count people entering a region or detect abandoned bags—each instance needs its own box across frames.
- **Safety**: flag workers near heavy machinery by detecting *instances* and their proximity, not just “person somewhere.”

Detection is a **bridge task**: more spatial than whole-image classification, but coarser than segmentation—boxes are cheap to annotate and fast to predict at scale.

### Main idea

In object detection, the model predicts a **set** of objects (variable length per image):

- A **class** for each detection.
- A **bounding box** (e.g., corners or center + size) in image coordinates.
- A **confidence** (or class probability) for filtering and ranking.

Useful pipeline intuition:

1. Extract visual features from the image.
2. **Propose** candidate regions (explicit region proposals, dense anchors, or implicit queries).
3. **Classify** each candidate and **regress** its box to hug the object.
4. **Deduplicate** so one real object does not produce many overlapping boxes (NMS for many architectures, or set-based training for DETR-style models).

Two common paradigms:

- **Two-stage detectors** (e.g., Faster R-CNN): first propose regions, then classify and refine each region—strong accuracy, more moving parts.
- **One-stage detectors** (e.g., YOLO): predict boxes and classes in one forward pass—very fast, popular for deployment.

**DETR** adds a third style: treat detection as **set prediction** with a transformer—no hand-designed anchors; a fixed set of learned **object queries** competes to explain the scene, and training uses **Hungarian matching** to assign predictions to ground truth.

Concrete scene: a busy crosswalk might contain several pedestrians, two cars, and a traffic light. The model must emit multiple $(\text{class}, \text{box})$ pairs, not one label for the whole image and not a per-pixel map.

```{figure} https://miro.medium.com/v2/resize:fit:1400/1*wZpyZlsX12LVFRjtiVsXkQ.png
:width: 80%
:alt: Detection-style visual prediction pipeline

Object detection predicts categories and bounding boxes for multiple objects in one image.
```

### Architecture

We will focus on **DETR** for object detection, with brief context for two-stage and one-stage families.

#### DETR (Detection Transformer)

DETR reframes detection:

- A **backbone** (CNN or ViT) extracts image features.
- A **transformer** encoder/decoder reasons globally over the scene.
- A fixed set of learned **object queries** each asks, in effect: “should I claim an object, and if so, what class and box?”
- Each query outputs:
  - Class probabilities (including a **no-object** / background slot),
  - One bounding box.

Key intuition:

- **No anchor grids** and no handcrafted proposal pipeline in the classical sense.
- The model learns to emit a **set** of objects; unused queries settle on the no-object class.
- **Hungarian matching** during training pairs predictions with ground-truth objects one-to-one so each GT box is matched to exactly one prediction and the loss is well defined.

Queries are not guaranteed to be “interpretable slots” (query 1 = person), but in practice different queries often specialize toward typical sizes or locations in the training distribution.

#### How DETR compares to Faster R-CNN and YOLO

- **Faster R-CNN**
  - Strong accuracy and a mature ecosystem.
  - More pipeline complexity (RPN, ROI align, per-ROI heads, often NMS).
- **YOLO**
  - Very fast and practical for real-time and edge devices.
  - Strong engineering around speed and export (TensorRT, mobile).
- **DETR**
  - Cleaner mathematical story and global reasoning over the image.
  - Historically slower to converge; variants such as **Deformable DETR** address attention cost and training stability.

### Training

A practical detection training pipeline:

1. **Dataset and annotations**
   - Ensure boxes are valid ($x_{\min} < x_{\max}$ and $y_{\min} < y_{\max}$) and use a consistent format (e.g., $xyxy$ vs.\ $cxcywh$).
   - Keep class IDs aligned across train/val/test; reserve background handling as your framework expects.
2. **Preprocessing**
   - Resize or letterbox images; **scale and shift box coordinates** in lockstep with the image transform.
   - Normalize with backbone-specific mean and standard deviation.
3. **Augmentation**
   - Random resize, crop, flip; avoid crops that drop all objects unless your task allows empty images.
4. **Optimization**
   - AdamW is common for DETR-style models.
   - Lower learning rate on the pretrained backbone than on the detection head often improves stability.
5. **Validation**
   - Track **mAP** (or AP at fixed IoU), not only training loss.

What students should watch:

- **Small objects** are often hardest; report AP broken down by object size if the dataset provides it.
- **Class imbalance** can make frequent classes dominate the loss; check per-class AP.
- **Duplicate boxes** or **missed objects** often point to score thresholding, NMS settings, or insufficient training for rare classes.

#### Finetuning

For class projects, fine-tuning a **pretrained** detector is usually the best default:

1. Start from a checkpoint trained on COCO (or similar).
2. Resize/replace the classification head for your number of classes (including background semantics).
3. Optionally **warm-start** the head with the backbone frozen for a few epochs.
4. Unfreeze and fine-tune end-to-end, with a smaller learning rate on the backbone.

Why this works:

- Pretrained features already encode objects and rough location cues.
- You spend optimization budget adapting to your label set and imaging conditions rather than learning low-level vision from scratch.

### Evaluation

Detection quality depends on **both** getting the **class right** and placing the **box** so it overlaps the true object. Metrics are built from **IoU-based matching** between predictions and ground truth.

#### Matching predictions to ground truth

At evaluation time, predictions are usually sorted by **confidence** (high to low). For each class separately, a common procedure is:

1. Take the highest-scoring prediction; try to match it to an unmatched ground-truth box of that class with **IoU $\geq$ threshold** (e.g., 0.5).
2. If matched, count a **true positive** (TP); otherwise **false positive** (FP).
3. Repeat for the next prediction. Ground-truth boxes that are never matched are **false negatives** (FN).

So a prediction is not “correct” on its own—it must match the **right object** with **enough overlap**.

```{figure} https://upload.wikimedia.org/wikipedia/commons/c/c7/Intersection_over_Union_-_visual_equation.png
:width: 70%
:alt: IoU equals intersection over union

IoU measures overlap between predicted and ground-truth boxes (same formula as for masks, but evaluated on rectangle areas). Matching uses IoU thresholds such as 0.5 or 0.75.
```

#### Precision, recall, and Average Precision (AP)

For a fixed IoU threshold, **precision** asks: among predicted boxes of this class, what fraction are correct? **Recall** asks: among all ground-truth boxes of this class, what fraction did we find?

Varying the **confidence threshold** traces out a **precision–recall curve**. **Average Precision (AP)** summarizes that curve (area under the PR curve in the usual definition used in detection benchmarks).

```{figure} https://upload.wikimedia.org/wikipedia/commons/2/26/Precisionrecall.svg
:width: 72%
:alt: Precision-recall curve with high and low threshold points

Raising the score threshold typically increases precision but lowers recall; AP integrates performance across operating points.
```

#### mAP and COCO-style reporting

- **AP@0.50** (often written **AP50**): IoU $\geq 0.5$ for a match—relatively forgiving localization.
- **AP@0.75** (**AP75**): IoU $\geq 0.75$—stricter box quality.
- **COCO mAP**: mean AP over **classes**, averaged over IoU thresholds from **0.50 to 0.95** in steps of 0.05—a single number that rewards both detection and tight localization.

Also report **per-class AP** and, when available, **AP for small / medium / large** objects—scalar mAP can hide failures on tiny or rare categories.

#### How to read metrics in practice

| Signal | What it suggests |
|--------|------------------|
| High AP50, low AP75 | Finds objects but **sloppy boxes**—improve regression or training augmentation. |
| Low AP on small objects only | **Resolution** or **feature stride** may be too coarse; consider multi-scale training or architectures with finer feature maps. |
| Good val loss, bad mAP | **Threshold / NMS** or **class calibration** may be off; inspect PR curves and score distributions. |

Numbers alone miss **systematic errors** (e.g., confusion between “truck” and “bus”). Always inspect **qualitative** results: true positives, false positives, and missed objects.

### Math formulation summary

Let the model produce $N$ predictions indexed by $i$, each consisting of box parameters $\hat{b}_i$ and class distribution $\hat{p}_i$ over $C$ foreground classes plus background/no-object:

$$
\{(\hat{b}_i, \hat{p}_i)\}_{i=1}^{N}.
$$

In DETR-style training, a **Hungarian algorithm** finds the minimum-cost bipartite matching between the $N$ predictions and the $M$ ground-truth objects (padding with “no object” as needed), so each GT is assigned at most one prediction.

A simplified **detection loss** combines classification and box terms:

$$
\mathcal{L}
=
\lambda_{\text{cls}}\mathcal{L}_{\text{cls}}
+
\lambda_{\text{box}}\mathcal{L}_{\text{box}}
+
\lambda_{\text{iou}}\mathcal{L}_{\text{iou}}.
$$

Typical terms:

- $\mathcal{L}_{\text{cls}}$: classification loss (e.g., cross-entropy on matched pairs).
- $\mathcal{L}_{\text{box}}$: regression loss on box parameters (e.g., $\ell_1$ on normalized coordinates).
- $\mathcal{L}_{\text{iou}}$: **GIoU** / **DIoU** / **CIoU**-style losses that align predicted boxes with ground truth beyond axis-aligned $\ell_1$ alone.

At **evaluation** time, AP is computed from **ranked** predictions using IoU-based matching (as in the Evaluation section)—the training loss and the ranking metric are related but not identical.

### Starter code

Below is a minimal PyTorch-style scaffold for the training loop with a torchvision DETR-style model:

- **Inputs**: lists of images and target dicts (`boxes`, `labels`).
- **Training**: the model returns a **loss dictionary**; sum its values for backprop.
- **Inference**: filter by **score threshold**; real deployments often add NMS depending on the architecture.

It is intentionally simplified for teaching and does not expose full DETR internals.

```python
import torch
import torch.nn as nn
from torchvision.models.detection import detr_resnet50


class Detector(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # num_classes includes background/no-object handling depending on implementation
        self.model = detr_resnet50(weights="DEFAULT")
        in_features = self.model.class_embed.in_features
        self.model.class_embed = nn.Linear(in_features, num_classes)

    def forward(self, images, targets=None):
        """
        images: list of tensors, each (3, H, W)
        targets: list of dicts with keys:
            - boxes: FloatTensor [num_boxes, 4] in xyxy format
            - labels: LongTensor [num_boxes]
        """
        return self.model(images, targets)


def train_one_step(model, optimizer, images, targets):
    """
    images: list[Tensor], targets: list[dict]
    """
    model.train()
    loss_dict = model(images, targets)  # torchvision detection models return dict of losses
    total_loss = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return {k: v.item() for k, v in loss_dict.items()}, total_loss.item()


@torch.no_grad()
def infer(model, images, score_thresh=0.5):
    model.eval()
    outputs = model(images)  # list of predictions

    filtered = []
    for pred in outputs:
        keep = pred["scores"] >= score_thresh
        filtered.append(
            {
                "boxes": pred["boxes"][keep],
                "labels": pred["labels"][keep],
                "scores": pred["scores"][keep],
            }
        )
    return filtered
```

Suggested student exercises:

1. Fine-tune on a small custom dataset and report AP@0.50 and COCO-style mAP if tooling allows.
2. Sweep confidence threshold and plot precision vs recall (or F1) on validation data.
3. Analyze failure cases by object size and occlusion.
4. Try a YOLO baseline and compare latency and accuracy against DETR on the same hardware.

### Advanced reading

- [DETR paper (Facebook Research)](https://arxiv.org/abs/2005.12872)
- [Deformable DETR](https://arxiv.org/abs/2010.04159)
- [YOLOv8 docs](https://docs.ultralytics.com/)

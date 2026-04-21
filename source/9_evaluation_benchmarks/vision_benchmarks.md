## Vision and multimodal benchmarks

### What a benchmark bundle contains

A **benchmark** usually packages:

1. **Task definition** (input, output, allowed training data)
2. **Dataset** with **annotations** and **license**
3. **Metric** and **evaluation server** or reference implementation
4. **Leaderboard** (optional) with **submission rules**

Comparing two papers is only fair when **training budgets**, **extra data**, and **ensemble** rules match the benchmark’s **track**.

---

### Classic image-level and object benchmarks

| Benchmark | Stresses | Notes |
|-----------|----------|--------|
| **ImageNet-1K** | Large-scale **classification** | Historic pretraining target; many variants (v2, ReaL labels) address **noise** in original labels |
| **COCO** | **Detection**, **segmentation**, **keypoints**, **captions** | mAP and mask AP are standard; drives detector design |
| **PASCAL VOC** | Detection / segmentation (older) | Smaller; still used pedagogically |
| **ADE20K** | **Semantic segmentation** in diverse scenes | Many fine-grained classes |
| **Cityscapes** | **Street** segmentation / depth | Domain-specific; high-quality annotations |

These datasets shaped **architecture** choices (ResNet, FPN, Mask R-CNN) and **pretraining** norms for years.

---

### Robustness, distribution shift, and “real world” suites

Academic in-distribution accuracy can **inflate** perceived capability. Benchmarks that stress **shift** include:

- **ImageNet-C / A / R / Sketch** — corruptions, adversarial filters, renditions, sketch domain
- **ObjectNet** — controlled viewpoint and clutter shifts
- **VTAB** (Vision Task Adaptation Benchmark) — **transfer** to many tasks with fixed adaptation protocol
- **WILDS** — subpopulations and **distribution shift** across domains (camera, hospital, location)

Use these when the product must work **off the training manifold**, not only on a clean test split.

---

### Video, 3D, and embodied evaluation

- **Kinetics / Something-Something** — **action recognition** and temporal reasoning
- **DAVIS / YouTube-VOS** — **video object segmentation**
- **ScanNet / nuScenes** — **3D** indoor / driving perception with **LiDAR** or **multi-view** constraints

Evaluation here adds **temporal consistency**, **tracking** identity, and **sensor** calibration issues.

---

### Multimodal and “foundation model” benchmarks

Vision-language and general **VLM** leaderboards (examples touched elsewhere in the book) include **VQA**, **GQA**, **DocVQA**, **TextVQA**, **ChartQA**, **MMMU**, **MathVista**, and **MM-Vet**-style rubrics. They differ along:

- **Input:** single image vs **multi-image** vs **video**
- **Output:** short answer vs **long-form** reasoning vs **JSON** tool calls
- **Grounding:** whether spatial **boxes** or **evidence** are scored

When reading a **number**, always check: **zero-shot vs fine-tuned**, **model size**, **chain-of-thought** allowed or not, and **private** vs **public** test.

**Takeaway:** benchmarks are **contracts**. Choose suites that **match your risk**: in-distribution accuracy for sanity checks, **shift** and **task-specific** suites for deployment claims, and **multimodal** benchmarks when language and vision must **align**.

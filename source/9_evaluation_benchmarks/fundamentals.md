## Evaluation fundamentals

### Why evaluation is its own discipline

A model is only as trustworthy as the **protocol** used to measure it. In computer vision, **datasets**, **metrics**, **preprocessing**, and **split design** can change rankings dramatically. Treating “accuracy on a leaderboard” as ground truth without reading the **task definition** is one of the most common mistakes in applied ML.

This section sets vocabulary shared across **`vision_benchmarks.md`** and **`comparison.md`**.

---

### Data splits and leakage

**Train / validation / test:** you fit parameters on **train**, tune hyperparameters and early-stop on **validation**, and report **generalization** on **test** once—or use nested cross-validation when data are scarce.

**Leakage** appears when information from the **test distribution** influences training or selection:

- **Duplicate** or near-duplicate images across splits (common in web-scraped data)
- **Same scene** in train and test with different crops
- **Tuning on test** by repeatedly publishing results on the same benchmark until numbers look good

**Mitigation:** deduplication hashes, **group splits** (e.g. by patient ID in medical imaging, by camera in autonomous driving), and holding out a **private** test set for final claims.

---

### Metrics for common vision tasks

| Task family | Typical metric | What it rewards |
|-------------|----------------|-----------------|
| **Classification** | Top-1 / Top-5 **accuracy**, balanced accuracy | Correct label among classes |
| **Multi-label** | mAP per class, F1 | Partial correctness when several labels apply |
| **Object detection** | **mAP** at IoU thresholds (e.g. COCO-style AP@[.5:.95]) | Localization + classification jointly |
| **Instance / semantic segmentation** | **mIoU** (mean IoU), pixel accuracy | Region overlap with ground truth |
| **Depth / regression** | RMSE, δ thresholds, SILog | Error on continuous outputs |

**Intersection over Union (IoU)** for a predicted box or mask:

$$
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}.
$$

**Average precision (AP)** summarizes a precision–recall curve; **mAP** averages AP over classes or IoU settings. Detection benchmarks therefore punish both **missed objects** and **sloppy boxes**.

---

### Beyond a single scalar

One number rarely captures deployment needs:

- **Calibration:** are predicted **probabilities** meaningful for thresholds or rejection?
- **Robustness:** performance under **noise**, **blur**, **compression**, **domain shift**
- **Efficiency:** **latency**, **memory**, **energy** on the hardware you ship
- **Fairness:** **error rates** across subgroups (skin tone, geography, rare categories)

**Takeaway:** evaluation is **not** “compute accuracy once.” It is an **experimental design** problem: define the **population** you care about, choose **metrics** that align with decisions, and guard against **leakage** and **overfitting to public leaderboards**.

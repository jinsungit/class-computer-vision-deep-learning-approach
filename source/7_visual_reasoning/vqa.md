## Visual question answering (VQA)

### What is VQA?

**Visual question answering** pairs an **image** with a **natural-language question** and asks a model to produce a **short answer** (often a word or phrase, sometimes yes/no). The task sits at the intersection of **computer vision** and **natural language processing**: the model must **ground** linguistic concepts in pixels (objects, attributes, counts, spatial relations, activities) while also handling **syntax** and **world knowledge** implied by the question.

Typical inputs and outputs:

| Input | Output (examples) |
|-------|---------------------|
| Image + “What color is the umbrella?” | “Red” |
| Image + “How many dogs are visible?” | “Two” |
| Image + “Is it raining?” | “Yes” |

**Why it became popular:** VQA offers a **concrete benchmark** for “understanding” images beyond fixed label sets (ImageNet-style classification). It also exposes **failure modes**—models can exploit **dataset biases** or **question-only** shortcuts if evaluation is not careful.

---

### Datasets and evaluation

```{figure} https://upload.wikimedia.org/wikipedia/commons/0/01/COCO-Stuff-10K-examples.png
:width: 92%
:alt: Example COCO images with dense annotations

**COCO-style scenes for VQA:** real-world images with many overlapping objects (people, animals, stuff) make it possible to ask diverse questions about **objects**, **counts**, **attributes**, and **relations**. *Image credit: MS COCO dataset, illustration adapted from the COCO-Stuff subset page on Wikipedia.*
```

**Early influential benchmarks** include **DAQUAR** (indoor scenes, restricted answer vocabulary) and the **COCO-VQA** line (**VQA v1 / v2**), which scale to open-ended answers over diverse COCO images. Answers are often **aggregated** from multiple human annotators; evaluation uses **accuracy** with **soft consensus** (an answer is correct if it matches at least *n* annotators) to tolerate synonyms (“sofa” vs “couch”).

**Synthetic** datasets such as **CLEVR** later stressed **compositional reasoning** (counting, comparing attributes, relational chains) with **programmatic** scene generation so that shortcuts from real-world statistics are reduced.

For this course, remember two evaluation lessons:

1. **Question-only baselines:** train on questions *without* images; if accuracy is surprisingly high, the split may carry **language priors** (e.g. “Is there a …?” often yes for common objects).
2. **Answer distribution:** frequent answers (“yes”, “no”, colors) can dominate; **balanced** re-sampling or **debiased** splits (as in VQA v2) try to reduce **cheating** via marginal statistics.

---

### Early pipeline: CNN image encoder + RNN question encoder

The dominant **first-generation** architecture (roughly mid-2010s) treated VQA as **late fusion** of two fixed-length vectors:

1. **Image branch:** a **CNN** (often VGG or ResNet pretrained on ImageNet) maps the image to a **global** feature vector (sometimes from the last pooling layer before the classifier).
2. **Question branch:** words are embedded (one-hot or learned embeddings), then fed to an **LSTM** or **GRU**; the **last hidden state** (or a pooled state) summarizes the question.
3. **Fusion + classifier:** concatenate or element-wise **product** of image and question vectors, then a **multi-layer perceptron** (MLP) outputs scores over a **fixed vocabulary** of candidate answers (the top *K* most frequent answers in training).

```{mermaid}
flowchart LR
  img[Image] --> cnn[CNN encoder]
  q[Question tokens] --> emb[Embeddings]
  emb --> rnn[LSTM or GRU]
  cnn --> fv[Image vector]
  rnn --> qv[Question vector]
  fv --> fuse[Fusion MLP]
  qv --> fuse
  fuse --> ans[Answer logits]
```

**Intuition:** the CNN finds “what is in the picture” at a coarse level; the RNN finds “what is being asked”; the MLP learns **compatibilities** (e.g. red umbrella + “color” questions).

**Strengths:** simple, fast to implement, strong **baseline** when pretrained CNNs are used.

**Limitations:**

- **Global vector bottleneck:** a single vector may lose **spatial** detail (which object is “left” vs “right”).
- **Shallow interaction:** one fusion step is weak for **multi-step** reasoning (“the object behind the red one”).
- **Fixed answer set** misses rare phrasings unless the vocabulary is huge.

---

### Attention and stronger interaction

```{figure} https://upload.wikimedia.org/wikipedia/commons/9/96/Attention-mechanism-in-neural-machine-translation.png
:width: 88%
:alt: Sketch of an attention mechanism highlighting different parts of an input

**Attention over image regions:** in VQA the model computes attention weights over **spatial feature maps** or **object proposals**, focusing on different regions depending on the question. The idea is analogous to attention in sequence-to-sequence models, shown here for text. *Image: Bahdanau-style attention diagram, adapted from the attention article on Wikipedia (schematic only).*
```

To let the model **look at different image regions** depending on the question, **attention** mechanisms became standard:

- **Soft attention:** weighted sum of **spatial** CNN features, where weights depend on the question (and sometimes previous attention states). **Stacked attention networks (SAN)** are a representative line of work.
- **Bottom-up and top-down attention:** “bottom-up” proposals from **region features** (e.g. Faster R-CNN) supply candidates; “top-down” question conditioning **selects** relevant regions—useful when objects are small or cluttered.

These models still often **predict over a trimmed answer set**, but they significantly improve **grounding** compared to a single global pool.

---

### Beyond single-turn classification

Later work extended the same ideas toward **dialog**, **referring expressions**, and **explanations**, but the **core VQA formulation** above remains the reference point for **benchmarking** vision-language fusion before the era of large **multimodal** transformers (covered in `visual_reasoning.md`).

**Takeaway:** early VQA showed that **deep fusion** of vision and language is learnable, but **evaluation design** and **inductive biases** (attention, regions) matter as much as raw model capacity for reliable “understanding” metrics.

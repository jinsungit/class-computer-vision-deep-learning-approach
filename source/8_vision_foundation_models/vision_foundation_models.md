## Vision foundation models

### From ImageNet specialists to general visual backbones

**Vision foundation models** are large models trained on **wide** visual data so they transfer to **detection, segmentation, depth, retrieval, video, documents**, and **vision-language** tasks with minimal task-specific architecture changes.

**Representative families** (briefly—many others exist):

| Model / family | Pretraining flavor | Typical use |
|----------------|-------------------|-------------|
| **CLIP** / **SigLIP** | Image–text **contrastive** alignment | Zero-shot classification, retrieval, text-conditioned vision |
| **DINOv2** | **Self-supervised** ViT on images | Dense features, geometric / correspondence tasks |
| **SAM** (Segment Anything) | Interactive **segmentation** at scale | Promptable masks, labeling pipelines |

These illustrate two ideas: **alignment to language** (CLIP-style) versus **pure vision representation learning** (DINO-style), often **combined** in modern multimodal systems.

---

### Qwen3-VL as a case study in unified multimodal design

**Qwen3-VL** is a recent **open** vision-language model family from the Qwen team: one stack handles **images, video, long documents, OCR, GUI/agent** scenarios, and **text** on par with strong text-only models. Official materials include the GitHub repository [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), model weights on **Hugging Face** and **ModelScope**, and the technical report [Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/pdf/2511.21631).

#### Model design (architecture)

The public README and paper highlight three **architecture** themes:

1. **Interleaved-MRoPE** — Multimodal **rotary positional** structure that allocates positional capacity across **time, width, and height**, improving **long-horizon video** and spatial structure compared with treating video as a flat token sequence.
2. **DeepStack** — Feeds **multi-level ViT** features (shallow to deep) into the language backbone so fine **texture and layout** signals are not collapsed too early; sharpens **image–text alignment** and fine-grained tasks (OCR, small objects).
3. **Text–timestamp alignment** — For video, aligns language with **explicit timestamps** for stronger **temporal grounding** (events localized in time), beyond coarse frame ordering alone.

**Backbone sizes:** the family includes **dense** checkpoints (e.g. 2B / 4B / 8B / 32B parameters) and **mixture-of-experts (MoE)** variants (e.g. 30B-A3B, 235B-A22B naming reflects total vs active parameters in MoE style releases). There are **Instruct** checkpoints (chat, following instructions) and **Thinking** variants optimized for **longer reasoning traces** before answers—parallel to text-only “reasoning” editions.

Conceptually, the pipeline matches other **ViT + LLM** systems: a **vision encoder** turns images or video frames into **tokens**; a **large language model** attends over text and visual tokens **jointly**. Qwen3-VL’s differentiators are in **positional encoding**, **multi-scale vision fusion**, and **video–text alignment** details above.

#### Training process (high level)

Exact mixture weights and data lists are documented in the **technical report**; at course level, the story is:

- **Stage 1 — Large-scale alignment:** train or continue-train the vision tower and projection layers on **massive image–text and video–text** data (web-scale pairs, captions, OCR-heavy documents, synthetic tasks) so visual tokens match the **token economy** of the LLM.
- **Stage 2 — Multimodal continuation:** interleave **text-only** and **multimodal** batches so the base model **does not forget** language: the report emphasizes **balancing objectives** (e.g. reweighting schemes so multimodal gradients do not wash out linguistic capability).
- **Stage 3 — Instruction tuning:** curated **instruction–response** data (dialogue, tools, grounding formats, agents, math with figures) to make outputs **useful and safe** in assistants.
- **Optional reasoning / RL stages:** **Thinking** editions add training that favors **extended chains** and stronger **STEM / code** behavior, similar in spirit to text reasoning models.

Students who need implementation detail (losses, data filtering, exact schedules) should read **arXiv:2511.21631** and the training sections of the README.

#### Where to get models, code, and data

- **Weights:** [Hugging Face Qwen organization](https://huggingface.co/Qwen) and [ModelScope Qwen3-VL collection](https://modelscope.cn/collections) host Instruct, Thinking, and some **quantized** (e.g. FP8) builds for deployment.
- **Code & cookbooks:** [github.com/QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) includes inference examples (e.g. `transformers>=4.57.0` in their quickstart), **Colab cookbooks** (OCR, document parsing, video, 2D/3D grounding, computer-use / mobile agents, multimodal coding).
- **Training data:** the **full training corpus is not shipped** as a single public download; the paper describes **sources and curation** (web image–text, video, PDFs, synthetic generations, licensed partners, etc.). For **your** applications, you typically add **private** data via **`downstream.md`** fine-tuning rather than reproducing pretraining.

**Takeaway:** Qwen3-VL is a useful **reference architecture** for how modern **vision-language foundation** models combine **ViT features**, **long-context LLMs**, **video positional structure**, and **instruction-heavy** post-training—while weights and tooling remain **open** for research and teaching.

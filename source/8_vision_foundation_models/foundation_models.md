## Foundation models

### What “foundation model” means

A **foundation model** is a large neural network trained **once**, at scale, on **broad** data (often web-scale text, images, or both). The same weights are then **reused** for many downstream problems through **fine-tuning**, **prompting**, **retrieval**, or **lightweight adapters**—instead of training a separate specialist model from scratch for every task.

The term was popularized in machine learning policy and research discussions (e.g. the Stanford Center for Research on Foundation Models report) to highlight a shift in engineering:

| Traditional pipeline | Foundation-model pipeline |
|----------------------|---------------------------|
| Task-specific dataset → train from random init | Massive pretraining → **adapt** to each task |
| Narrow distribution | **Heterogeneous** pretraining mix |
| One model per application | One (or few) **backbones** for many applications |

**Key properties** people usually emphasize:

1. **Scale:** parameters, data, and compute are large enough that **transfer** to new tasks is strong.
2. **Generality:** the pretraining objective (next-token prediction, contrastive alignment, masked modeling, etc.) is not identical to any single deployment task, yet produces **broad** capabilities.
3. **Adaptability:** the model can be steered with **natural language instructions**, **few-shot examples**, or **parameter-efficient** updates.

---

### Large language models as the canonical example

**Large language models (LLMs)** are the clearest instance of foundation models today. A single **decoder-only transformer** (or mixture-of-experts variant) pretrained on diverse text learns:

- **Syntax and semantics** over many languages and styles
- **World knowledge** (fragile but useful) and **reasoning-like** patterns
- **Instruction following** after alignment stages (supervised fine-tuning, preference optimization, etc.)

From one base checkpoint, practitioners routinely derive:

- **Chat assistants** (dialog formatting, safety, tone)
- **Code models** (continue pretraining or fine-tune on code)
- **Math or tool-use** variants (RL with verifiers, tool APIs)
- **Domain experts** (law, medicine, biology) via continued pretraining or LoRA on proprietary corpora

**In-context learning** is an important special case of “adaptation without weight updates”: the model reads a **prompt** (instructions + a handful of labeled examples) and **generalizes** to new labels or formats at inference time. That is not a substitute for **reliable** deployment on critical tasks, but it explains why one checkpoint can cover **many** language tasks in a product surface.

---

### Why this matters for computer vision

Vision has moved in the same direction: **image encoders** and **vision-language** systems are pretrained on large mixtures (image–text pairs, video, documents, UI screens), then **specialized** with small task-specific data. The rest of this chapter connects that story: **`vision_foundation_models.md`** focuses on a concrete multimodal family (**Qwen3-VL**), and **`downstream.md`** discusses how you **adapt** foundation models in practice.

**Takeaway:** a foundation model is less a single “magic algorithm” than an **economic and scientific** bet—**amortize** huge training cost across many tasks, then pay a **small** adaptation cost per application.

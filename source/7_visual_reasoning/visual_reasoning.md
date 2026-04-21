## Visual reasoning: tasks, models, and benchmarks

### What changes when reasoning is visual?

```{figure} https://upload.wikimedia.org/wikipedia/commons/3/3a/Relational_reasoning_example_in_scene_graphs.png
:width: 88%
:alt: Example scene graph with objects and relations

**From objects to relations:** visual reasoning often requires modeling **relations** between objects (left of, behind, larger than) rather than just detecting them individually. Scene graphs and relational diagrams like this one are a useful way to think about the structure that a model implicitly needs to learn. *Image adapted from scene-graph examples on Wikipedia (schematic).*
```

**Visual reasoning** asks for **multi-step** conclusions grounded in **non-textual** signals: photographs, synthetic diagrams, **charts**, **geometric** figures, screenshots of **software**, or **documents** with layout. Unlike classic VQA (`vqa.md`), which often maps to a **short answer** with relatively local evidence, visual reasoning stress-tests **composition**, **quantitative** reading of plots, **spatial** logic, and **cross-region** integration—sometimes overlapping with **document AI** and **GUI automation**.

A useful mental model:

| Axis | Classic VQA (early) | Visual reasoning emphasis |
|------|---------------------|---------------------------|
| Evidence | Often one salient region | Multiple regions, **relations**, or **numeric** trends |
| Modality | RGB image | Charts, schematics, **multi-image** comparisons |
| Output | Word from fixed set | **Long-form** text, **programs**, or **actions** |

---

### Approaches: from adapters to native multimodal training

```{figure} https://upload.wikimedia.org/wikipedia/commons/2/25/Vision_Transformer_schematic.png
:width: 82%
:alt: Schematic of a Vision Transformer that splits an image into patches and processes them with a transformer encoder

**Vision encoders for reasoning:** many modern visual reasoning systems feed **ViT-style** patch embeddings into a multimodal transformer. The text side sees tokens; the image side sees patch tokens; cross-attention allows the model to jointly reason. *Image credit: Vision Transformer schematic from Wikipedia (simplified).*
```

**Modular era (pre–large multimodal models):** CNN or ViT **encoder** + RNN/Transformer **decoder**, with **cross-attention** between image patches and text tokens. **BLIP**, **Flamingo**-style **frozen** encoders with lightweight **adapters** on a large LLM showed that **reuse** of text pretraining accelerates multimodal skill.

**Native multimodal LLMs:** models trained (at scale) on **interleaved** image-text data learn a **single** transformer over **visual tokens** (from a ViT or patchifier) and **text tokens**. Commercial systems (e.g. **GPT-4V**, **Gemini**) and open weights (e.g. **LLaVA**, **Qwen-VL**, **InternVL**) exemplify this: **instruction tuning** aligns behavior to **chat**, **tool use**, and **chain-of-thought** in the **same** interface as text-only assistants.

**Agents and GUIs:** recent **vision-language agents** treat a **screen** as observation and output **mouse/keyboard** or **API** actions, combining **visual reasoning** with **planning**—evaluation often uses **online** simulators or **recorded** trajectories.

---

### Benchmarks worth knowing

Below is a **non-exhaustive** list that captures different **skills**; papers move quickly, so treat scores as **snapshots** and read **task definitions** carefully (some require **external** tools; others are **multiple-choice** only).

| Benchmark | Stresses | Notes |
|-----------|----------|--------|
| **MMMU** | College-level **multimodal** questions (STEM, humanities) | Multi-image and **mixed** modalities per question |
| **MathVista** | **Math** + **figure** reasoning | Combines several prior math/visual sources |
| **ChartQA** | **Chart** reading and **aggregation** | Tests **layout** + **numeric** extraction |
| **DocVQA / InfographicVQA** | **Text in the wild**, diagrams | OCR + semantics |
| **ScienceQA** (image subset) | **Science** with diagrams | Mixed modality **rationale** in some versions |
| **V* benchmark** | **Long** videos, **fine-grained** recall | Stresses **memory** over time |
| **MM-Vet** | **Interleaved** capabilities in one rubric | Aggregates **subskills** with a single scoring protocol |

**Interesting research directions** visible in these suites:

- **Tool-augmented** visual reasoning (code execution, calculator) vs **closed-book** only.
- **Self-verification:** models re-read the image or **crop** before answering.
- **Hard negatives** that break **language-only** guessing—similar spirit to debiased VQA.

---

### Pedagogical connections

1. **Geometry chapter (Ch. 5):** depth and 3D are **visual reasoning** with **continuous** outputs; here the emphasis is often **discrete** decisions or **language**.
2. **Generation chapter (Ch. 6):** understanding **diffusion** or **token** models helps when multimodal systems **generate** or **edit** images conditioned on **plans** expressed in text.

**Takeaway:** visual reasoning is where **vision encoders**, **LLM reasoning habits** (`reasoning.md`), and **careful benchmarks** meet. Progress is measured not only by **accuracy** but by **robustness** to **layout**, **chart junk**, and **multi-step** evidence—exactly where **shortcut learning** from the VQA era reappears unless datasets and rewards are designed with care.

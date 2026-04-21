## From foundation models to applications

### What you can build on top

A pretrained **foundation model** is rarely the final product. Most teams **compose** it with:

- **Prompts and system policies** (tone, format, safety refusals)
- **Retrieval (RAG)** over private documents or a vector database
- **Tools** (calculator, code execution, search, CRM APIs)
- **Fine-tuning** or **adapters** when behavior must match a **narrow** domain

The same menu applies to **vision-language** checkpoints (e.g. Qwen3-VL Instruct): the backbone handles **perception and language**; your product adds **data, constraints, and evaluation**.

---

### Fine-tuning: when and how

**Full fine-tuning** updates all parameters. It can yield the best **task fit** but needs more **GPU memory**, risks **catastrophic forgetting** of general skills, and can **overfit** small datasets.

**Parameter-efficient fine-tuning (PEFT)**—especially **LoRA** / **QLoRA**—trains **low-rank** matrices injected into attention (and sometimes MLP) layers while freezing the base weights. Benefits:

- **Much smaller** GPU footprint (QLoRA loads 4-bit weights + adapter grads)
- **Cheaper iteration** when you have thousands—not billions—of labeled examples
- **Multiple adapters** for different customers or tasks on one base

**Instruction / chat fine-tuning** on **JSON or dialogue** formats teaches the model your **schema** (e.g. “always output bounding boxes as `[x1,y1,x2,y2]` in `[0,999]`”). **Preference optimization** (DPO, RLHF-style rewards) nudges **style** and **helpfulness** when scalar feedback exists.

For Qwen vision models, the Qwen team maintains **fine-tuning code** for earlier VL generations (see links from the [Qwen3-VL README](https://github.com/QwenLM/Qwen3-VL) to the `qwen-vl-finetune` path under Qwen2.5-VL); the **pattern**—multimodal JSON batches, image resizing, packing—is what transfers when you move to newer checkpoints.

---

### Customization without changing weights

Many deployments never **gradient-update** the base model:

| Technique | What you control | Typical cost |
|-----------|------------------|--------------|
| **System + user prompts** | Format, steps, tools | Engineering time |
| **Few-shot in context** | Per-task examples | Context length / latency |
| **RAG** | Factual grounding from your corpus | Indexing + retrieval quality |
| **Constrained decoding** | JSON grammar, valid APIs | Parser / runtime |

For **vision**, RAG might mean retrieving **similar images**, **product manuals**, or **past tickets** along with text chunks; the VLM **grounds** answers in both the **query image** and retrieved evidence.

---

### Practical checklist

1. **Start from the right checkpoint:** general **Instruct** vs **Thinking** (latency vs reasoning), **size** vs **latency**, **open** vs **API-only** for compliance.
2. **Freeze first, measure:** strong prompts + eval suite often beats a rushed LoRA on 500 examples.
3. **Own your eval:** academic benchmarks do not replace **your** failure modes (OCR on your forms, UI screenshots at your resolution).
4. **Safety and license:** foundation weights come with **licenses**; customer data in fine-tuning needs **DP** / **redaction** policies as appropriate.

**Takeaway:** foundation models **buy** general perception and language; **your** job is **adaptation**—data, prompts, retrieval, adapters, and **evaluation**—so the system is **reliable** on the tasks you actually ship.

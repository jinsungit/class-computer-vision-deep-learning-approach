## Long text label (captioning)

### Motivation

Image-level labels give **short** tags; detection and segmentation localize **entities** or **regions**.
Many applications need **natural language**: a full sentence (or paragraph) that describes what matters in the image.

**Image captioning** connects vision and language by generating such text conditioned on pixels.

Why this matters:

- **Accessibility**: produce **alt text** for screen readers—e.g., “Chart shows revenue rising 2020–2024” for a dashboard, not just “screenshot.”
- **Search and indexing**: store human-readable descriptions so users can query images with phrases, not only keywords.
- **Human–AI interaction**: assistants and robots can **report what they see** in plain language before acting.
- **Multimodal stack**: captioning is a stepping stone to **VQA**, **instruction following**, and **vision-language agents** that reason over images and text jointly.

Supervision is a **sequence**: the target is not a single class ID but tokens $w_1, \ldots, w_T$ (words or subwords). Many different wordings can describe the same scene equally well, which shapes both training and evaluation.

### Main idea

Given an image $x$ (and often a **prompt** such as “Describe this image in one sentence.”), the model generates a caption **one token at a time**:

- **Input**: image $x$, optional prompt tokens.
- **Output**: caption tokens $(w_1, w_2, \ldots, w_T)$ until an end-of-sequence token.

Intuition-first view:

1. A **vision encoder** maps the image to a tensor of visual features or visual **tokens**.
2. A **language model** (decoder-only or encoder–decoder) conditions on those features and on tokens generated so far.
3. At step $t$, the model predicts $P(w_t \mid w_{<t}, x, \text{prompt})$; sampling or search turns that distribution into a finished string.

This is the same **conditional language modeling** story as machine translation, except the “source language” is pixels instead of a sentence.

**Many captions are valid.** For a photo of a dog catching a frisbee, both “A dog catches a frisbee in a park” and “A brown dog leaps to grab a flying disc” can be correct. Training usually picks **one** reference (or a small set); evaluation must tolerate **paraphrase**, not only exact word match.

A good caption tends to be:

- **Grounded**: what it mentions is **actually visible** (objects, relations, salient actions).
- **Informative**: covers the **main** content, not only a vague “something in a room.”
- **Fluent**: reads like natural language.

```{figure} https://miro.medium.com/v2/resize:fit:1400/1*z-TEUQ9jfIa5iW8I6ssPew.PNG
:width: 80%
:alt: Image captioning encoder-decoder pipeline

Image captioning: visual features feed a language decoder that generates the caption sequentially.
```

### Architecture

Captioning has two historical layers; in practice **vision–language models (VLMs)** are now the default for new projects.

#### Classical encoder–decoder captioning

Early and mid-era systems paired a **CNN** (image encoder) with an **RNN or Transformer decoder** trained with **teacher forcing**: at each step, the decoder sees the **ground-truth** previous token during training. At inference, it conditions on **its own** previous predictions—hence **exposure bias**.

```{figure} https://upload.wikimedia.org/wikipedia/commons/c/c7/Seq2seq_RNN_encoder-decoder_with_attention_mechanism%2C_training.PNG
:width: 85%
:alt: Seq2seq encoder-decoder with attention, training phase

Classical seq2seq with **attention**: the decoder can look at different parts of the encoded image when predicting each word—same high-level idea as cross-attention in modern VLMs, with different backbones.
```

#### Modern vision–language models (VLMs)

Recent systems often **reuse a large pretrained LM** instead of a caption-only decoder:

- Encode the image into visual tokens with a **vision encoder** (ViT, SigLIP, etc.).
- Map visual tokens into the language model’s space via a **projection** or **adapter**, and optionally **cross-attention** layers.
- **Prompt** the model (“Describe this image…”) and generate the caption autoregressively.

Why this is the default now:

- **Fluency** and world knowledge come from the LM, not only from caption datasets.
- **One model** can caption, answer questions, and follow short instructions after light alignment tuning.
- **Parameter-efficient fine-tuning** (LoRA, adapters) fits class-scale GPU budgets better than full pretraining.

Representative families include BLIP-2, LLaVA-style models, Qwen-VL, and proprietary multimodal chat models.

### Training

A practical captioning / VLM fine-tuning pipeline:

1. **Data preparation**
   - Each image has one or more **reference** captions (COCO-style) or instruction–response pairs (instruction-tuning).
   - Reuse a **subword tokenizer** from the pretrained VLM; do not invent a new vocabulary from scratch unless you train from zero.
2. **Input–output formatting**
   - Follow the model’s **chat template** or caption prefix (e.g., `USER: <image>\nDescribe… ASSISTANT:`).
   - Concatenate prompt and target caption so the loss applies to the answer tokens.
3. **Loss**
   - **Causal next-token cross-entropy** on supervised positions.
   - Often **mask** prompt tokens in the labels so the loss does not penalize “predicting the question.”
4. **Optimization**
   - AdamW with cosine or linear decay; **lower LR** on the vision tower than on the projector/LM adapters when unfreezing both.
   - Prefer **LoRA / adapters** on the LM (and sometimes attention) for limited compute.
5. **Inference decoding**
   - **Greedy** decoding for speed; **beam search** or **nucleus sampling** for better quality or diversity.

What students should watch:

- **Exposure bias**: training sees teacher tokens; inference sees **model** tokens—errors can compound. Scheduled sampling, RL fine-tuning, and better pretraining mitigate but rarely eliminate the gap.
- **Generic captions** (“a person standing in a room”) often mean **weak grounding** or **limited supervision**; check data and prompts.
- **Hallucination**: objects or relations **not in the image**—tie-break with qualitative checks and, if possible, **object-level** audits.

#### Finetuning

For class projects, start from a **caption-capable or instruction-tuned** VLM:

1. Choose a checkpoint with a documented processor and chat format.
2. Train **LoRA/adapters** on the language side and often the **projector** first.
3. Unfreeze more of the vision encoder only if data and regularization justify it, with a **small** learning rate.

Benefits: strong baselines with **less data** and **less compute** than training encoders and decoders from scratch.

### Evaluation

Caption quality is **subjective** and **multi-dimensional**: the same image admits many correct wordings, and a fluent sentence can still be **wrong** or **incomplete**.

#### Automatic metrics (reference-based)

These scores compare the generated caption to **one or more human references** (same image). They are cheap but **imperfect**—treat them as diagnostics, not ground truth.

| Metric | What it roughly measures | Limitation |
|--------|---------------------------|------------|
| **BLEU** | $n$-gram **precision** vs references | Penalizes valid paraphrases; brittle to word choice. |
| **ROUGE-L** | Longest **common subsequence** | Rewards long substring overlap, not necessarily meaning. |
| **METEOR** | Aligns words with **stems/synonyms** | Better on paraphrase than BLEU, still surface-level. |
| **CIDEr** | **TF-IDF–weighted** $n$-gram consensus across annotators | Common on COCO; tuned for caption-style judgments. |
| **SPICE** | **Scene-graph**–style tuple match | Closer to “objects and relations,” depends on parser quality. |

**Concrete intuition**: reference is *“A dog catches a frisbee in the air.”* A model output *“A brown dog leaps for a flying disc outdoors.”* might score **low BLEU** (few shared $n$-grams) yet be **excellent** for a human reader—so **report several metrics** and always show **qualitative** examples.

#### Best practice

- Report **multiple** scores (e.g., BLEU-4, METEOR, CIDEr, SPICE) when tooling allows.
- Show **success and failure** cases: grounding errors, missed salient objects, and template-like generic text.
- If the course allows, add a **small human study** (relevance, fluency, completeness on a fixed rubric)—even dozens of ratings can reveal what automatic numbers miss.

#### Reference-free signals (optional)

**CLIPScore** and similar models score **image–text alignment** without a string match to references. Useful for **filtering** or **A/B** comparisons, but they do not guarantee **coverage** of everything important in the image.

### Math formulation summary

Captioning models implement a **chain rule** over tokens: the probability of a full caption given the image and prompt **factors** as a product of next-token probabilities:

$$
P(w_{1:T} \mid x, \text{prompt}) = \prod_{t=1}^{T} P(w_t \mid w_{<t}, x, \text{prompt}).
$$

Training minimizes **negative log-likelihood** (causal LM loss) on the ground-truth token sequence $w_{1:T}^*$—typically only on **caption** positions if the prompt is masked in the labels:

$$
\mathcal{L}_{\text{cap}} = -\sum_{t=1}^{T} \log P(w_t^* \mid w_{<t}^*, x, \text{prompt}).
$$

Interpretation:

- Each step is a **multi-class** prediction over the vocabulary (with logits from the language head).
- **Global** caption quality arises from a long sequence of **local** next-token decisions conditioned on the image and prior tokens.

### Starter code

Below is a compact **Hugging Face**–style scaffold:

- **Inference**: `generate` from a vision-to-sequence model with a text prompt.
- **Training**: one step of **supervised** next-token loss on `prompt + caption` (prompt masking in labels is left as an exercise).

Checkpoints such as `Salesforce/blip2-opt-2.7b` or LLaVA-style models differ in API details; adjust `model_id` and processor usage to match the model card.

```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

device = "cuda" if torch.cuda.is_available() else "cpu"

# Example checkpoints: Salesforce/blip2-opt-2.7b, llava-hf/llava-1.5-7b-hf (needs setup)
model_id = "Salesforce/blip2-opt-2.7b"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id).to(device)


@torch.no_grad()
def generate_caption(image, prompt="Describe this image in one sentence."):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=40)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption


def train_one_step(images, prompts, target_captions, optimizer):
    """
    images: list[PIL.Image]
    prompts: list[str]
    target_captions: list[str]
    """
    model.train()

    # Build supervised text targets (prompt + expected caption format)
    full_text = [f"{p} {c}" for p, c in zip(prompts, target_captions)]
    batch = processor(images=images, text=full_text, return_tensors="pt", padding=True).to(device)

    # Standard causal LM loss; for cleaner supervision, mask prompt tokens in labels.
    outputs = model(**batch, labels=batch["input_ids"])
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

Suggested student exercises:

1. Compare **greedy**, **beam**, and **sampling** decoding on the same validation images.
2. Add **LoRA** fine-tuning and compare quality vs full fine-tuning under a fixed GPU-hour budget.
3. Audit **hallucinated objects** (mentioned in caption but absent in image) alongside CIDEr/SPICE.
4. Prompt for **style-controlled** captions: short alt text, dense technical description, vs casual social caption.

### Advanced reading

- [BLIP: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2201.12086)
- [LLaVA (visual instruction tuning)](https://arxiv.org/abs/2304.08485)
- [Qwen2-VL technical report](https://arxiv.org/abs/2409.12191)

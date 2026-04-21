## Reasoning in large language models

### From pattern matching to explicit deliberation

```{figure} https://upload.wikimedia.org/wikipedia/commons/4/41/Transformer_Block_Diagram.svg
:width: 72%
:alt: Diagram of a transformer block with self-attention and feed-forward layers

**Transformer backbone for reasoning:** modern large language models are deep stacks of **transformer blocks** with self-attention and feed-forward layers. “Reasoning” behaviors emerge when these architectures are scaled and then **post-trained** to prefer multi-step chains of thought. *Image credit: transformer block illustration from Wikipedia (schematic).*
```

For many years, scaling **autoregressive** language models improved fluency and factual recall, but **multi-step** tasks (math proofs, long programs, logic puzzles) remained brittle. A practical shift was to treat inference as **search over thoughts**: instead of emitting only the final answer in one pass, the model generates **intermediate text**—a chain of deductions, checks, or subgoals—before concluding.

**Chain-of-thought (CoT) prompting** showed that simply asking the model to “think step by step” (with a few **in-context** examples) can unlock better performance on arithmetic and symbolic tasks **without** weight updates. That observation blurred the line between “prompt engineering” and **algorithmic** use of LLMs: the **context window** becomes a **scratchpad**.

---

### Test-time compute and “reasoning models”

```{mermaid}
flowchart TD
  q[Problem] --> gen1[Sample chain 1]
  q --> gen2[Sample chain 2]
  q --> gen3[Sample chain 3]
  gen1 --> check1[Check / score 1]
  gen2 --> check2[Check / score 2]
  gen3 --> check3[Check / score 3]
  check1 --> select[Select best answer]
  check2 --> select
  check3 --> select
```

**Test-time search:** instead of returning the first sample, reasoning-focused systems generate **multiple chains**, evaluate them with **self-consistency** or external tools, and then select an answer—trading **compute** for **robustness**.

**Test-time compute** refers to spending more **inference** effort—longer generations, **sampling** multiple chains, **self-consistency** voting, or **tree search**—to improve reliability. Recent **reasoning-oriented** systems (often trained with **reinforcement learning** on verifiable rewards, or **distillation** from stronger teachers) push this further: they are **optimized** to produce long, structured rationales on math, code, and logic benchmarks.

**DeepSeek-R1** (2025) is a prominent public example of this family: a **post-trained** model emphasizing **extended reasoning traces** before answers, with reports of strong scores on benchmarks such as **AIME** (competition math) and **Codeforces**-style coding when paired with appropriate decoding and infrastructure. The high-level lesson for the course is not any single leaderboard number, but the **training recipe**:

| Ingredient | Role |
|------------|------|
| **Base LLM** | Broad language and world knowledge |
| **RL / preference optimization** | Align multi-step outputs with **reward** signals (e.g. unit tests, symbolic checkers) |
| **Distillation** | Transfer “reasoning style” to smaller models for deployment |

**Interpretation caveat:** long chains **look** like human deliberation, but they can still **hallucinate** confident steps. **Tool use** (calculator, compiler, theorem prover) and **verification** remain important for **trustworthy** systems.

---

### How this connects to vision

Pure **text** reasoning benchmarks (GSM8K, MATH, HumanEval) measure **symbolic** competence. **Visual reasoning** (`visual_reasoning.md`) asks the same habits—decomposition, self-check, planning—to operate on **pixels, diagrams, and UI screenshots**. The explosion of **text** reasoning capability in models like DeepSeek is therefore a **direct precursor**: multimodal systems inherit both the **strengths** (long-horizon chains) and **weaknesses** (ungrounded steps) of their LLM backbones.

---

### Practical notes for builders and researchers

1. **Prompting vs fine-tuning:** CoT works **zero-shot** on strong models; smaller models may need **SFT** on rationales or **RL** with verifiers.
2. **Length vs signal:** verbose traces cost **latency** and **money**; **compression** and **early stopping** when confidence is high are active engineering topics.
3. **Safety:** “reasoning” models can still produce **harmful** content if rewards only optimize task success; **policy layers** and **refusal** training remain necessary.

**Takeaway:** modern LLM “reasoning” is best understood as **learned search and verification in language space**, amplified by scale and **post-training**. Visual agents and VQA systems increasingly **reuse** these text-native habits once images are encoded into the same **token stream**.

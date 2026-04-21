## Comparing models, leaderboards, and human judgment

### Why public leaderboards differ from product evaluation

**Leaderboards** compress messy reality into a **scalar** or small table. They are useful for **progress** and **rough ordering**, but they rarely encode:

- **Latency** and **cost** per query at your batch size
- **Safety** refusals, **toxicity**, or **copyright** behavior
- **Customization** (your RAG corpus, your camera optics)
- **Drift** when upstream data or APIs change

Treat leaderboards as **one signal** in a larger evaluation plan (`fundamentals.md`).

---

### Arena-style and human-preference evaluation

For generative and **open-ended** outputs (captions, answers, edits), **automatic metrics** (BLEU, CIDEr, ROUGE) correlate imperfectly with human quality. **Human studies** and **pairwise comparisons** (“which answer is better?”) remain important.

**LMArena** ([leaderboard](https://lmarena.ai/leaderboard/vision)) runs **blind** side-by-side battles between models on **vision-language** prompts; aggregate **Elo-style** ratings reflect **crowd preferences** under their sampling and voter pool. Vision-specific categories and methodology are described in [LMArena’s vision tasks announcement](https://news.lmarena.ai/re-introducing-vision-arena-categories/).

**Strengths:** captures **holistic** usefulness—helpfulness, formatting, grounding—when voters are honest and instructions are clear.

**Caveats:** voter **bias**, **gaming**, **exploitability** of the prompt distribution, and mismatch with **your** user demographics or **enterprise** requirements.

---

### Reproducibility checklist

When you compare two checkpoints **yourself**:

1. **Pin** library versions (CUDA, `torch`, `transformers`, dtype).
2. **Document** resolution, **letterbox** vs **crop**, and **normalization**—small preprocessing gaps move mAP.
3. **Use official** eval scripts when available; **batch size** can affect BatchNorm behavior in legacy CNNs.
4. **Report** compute: **FLOPs**, **throughput**, **memory**, not only accuracy.

---

### Building an internal benchmark

For teaching and for industry projects, the most reliable path is often a **private** “golden set”:

- A few hundred to a few thousand **curated** examples with **adjudicated** labels
- **Stratified** difficulty (easy / medium / hard failure modes you have seen in production)
- **Regression tests** run on every release

Pair this with **public** benchmarks for **external** comparability.

**Takeaway:** combine **standard benchmarks** for context, **stress tests** for robustness, and **human or arena-style** signals for open-ended quality—always tied back to the **decisions** your system will make in deployment.

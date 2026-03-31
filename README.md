# Computer Vision: The Deep Learning Approach — Online Book

Sphinx-based online documentation for **CSCI 4900/6900** (Spring 2026).  
Mixed text, figures, and code; structure follows the course outline from the introduction slides.

## Structure

- **`source/`** — Sphinx source (Markdown `.md` files via MyST, `conf.py`, `_static/`, `_templates/`)
- **`1.introduction.md`** — Original course intro/slides (reference)
- **`build/`** — Generated output (created when you build)

### Chapters (from outline)

| # | File | Topic |
|---|------|--------|
| 0 | `introduction.md` | Introduction |
| 1 | `chapter01_basics.md` | The basics (math, ML, DL, optimization, coding) |
| 2 | `chapter02_neural_nets.md` | Neural nets (CNN, U-Net, Transformers, ViT) |
| 3 | `chapter03_unsupervised.md` | Unsupervised & self-supervised |
| 4 | `chapter04_supervised_semantics.md` | Supervised with semantics |
| 5 | `chapter05_supervised_geometry.md` | Supervised with geometry |
| 6 | `chapter06_generation.md` | Generation |
| 7 | `chapter07_visual_reasoning.md` | Visual reasoning |
| 8 | `chapter08_foundation_models.md` | Vision foundation models |
| 9 | `chapter09_evaluation.md` | Evaluation and benchmarks |
| 10 | `chapter10_bias_ethics.md` | Bias and ethics |

## Setup and build

### 1. Install dependencies

Upgrade pip first (helps avoid building wheels from source on Windows):

```bash
python -m pip install --upgrade pip
pip install -r requirements-docs.txt
```

**If you see "Failed building wheel for MarkupSafe" on Windows:** the requirements pin `MarkupSafe>=2.1.1` so pip uses a pre-built wheel. Ensure pip is up to date; if it still fails, try: `pip install --only-binary :all: MarkupSafe` then `pip install -r requirements-docs.txt`.

### 2. Build HTML

From the ``source`` directory:

**Option A — Sphinx directly:**

```bash
cd source
sphinx-build -b html . _build/html
```

Or with Python module: ``python -m sphinx -b html . _build/html``

**Option B — Make (Linux/macOS):** ``make html``  

**Option C — Windows batch:** ``make.bat html``

### 3. View the site

Open `source/_build/html/index.html` in a browser.

### 4. Auto-rebuild while editing (optional)

```bash
pip install sphinx-autobuild
cd source
sphinx-autobuild . _build/html
```

Then open http://127.0.0.1:8000

## Adding content

All docs are **Markdown** (`.md`) using [MyST](https://myst-parser.readthedocs.io/). You can use standard Markdown plus Sphinx directives.

- **Text:** Edit the `.md` files in `source/`. Use normal Markdown: `#` headings, `**bold**`, `[text](url)`, lists, etc.
- **Figures:** Put images in `source/_static/imgs/` and use a MyST figure directive:

  ````markdown
  ```{figure} _static/imgs/your_image.png
  :width: 70%
  :alt: Short description

  Your caption here.
  ```
  ````

- **Code:** Use fenced code blocks with a language:

  ````markdown
  ```python
  import torch
  x = torch.randn(3, 4)
  ```
  ````

- **Math:** Use `$...$` for inline and `$$...$$` for block equations (MathJax is enabled).

## Customization

- **Theme:** `conf.py` uses `sphinx_rtd_theme`. Change `html_theme` to use another theme (e.g. `furo`, `pydata_sphinx_theme`).
- **Logo/favicon:** Set `html_logo` and `html_favicon` in `conf.py`.
- **PDF:** Install a LaTeX distribution and run `sphinx-build -b latex . _build/latex`, then build the PDF from the generated `.tex` files.

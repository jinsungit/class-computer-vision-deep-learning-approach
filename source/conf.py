# Configuration file for the Sphinx documentation builder.
# CSCI 4900/6900 Computer Vision: The Deep Learning Approach

import os
import sys

# -- Project information -----------------------------------------------------
project = "Computer Vision: The Deep Learning Approach"
copyright = "2026, University of Georgia, School of Computing"
author = "Jin Sun"
release = "Spring 2026"
version = "2026"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_copybutton",  # one-click copy for code blocks
]

# Treat .md files as MyST Markdown (so math and directives work)
source_suffix = {
    ".md": "myst",
    ".rst": "restructuredtext",
}

# Enable MyST math syntax (inline and block)
myst_enable_extensions = [
    "dollarmath",  # $...$, $$...$$
    "amsmath",     # \[...\], equation environments
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# html_theme = "sphinx_rtd_theme"  # pip install sphinx_rtd_theme
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_js_files = ["pathfix.js"]
html_css_files = ["custom.css"]
html_theme_options = {
    # Ensure the right-side "Contents" includes subsection headings (###).
    "show_toc_level": 3,
}
html_logo = None  # set to "_static/logo.png" if you have one
html_favicon = None

# Optional: custom CSS
# html_css_files = ["custom.css"]

# -- Options for code blocks -------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: |Out\[\d*\]: "
copybutton_remove_prompts = True

# -- Options for LaTeX/PDF ---------------------------------------------------
latex_elements = {}
latex_documents = [
    ("index", "cv-dl-book.tex", project, author, "manual"),
]

# -- Intersphinx (optional, for linking to Python/PyTorch docs) ---------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

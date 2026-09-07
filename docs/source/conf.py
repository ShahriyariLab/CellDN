"""Sphinx configuration for CellDN documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make src/celldn importable for API documentation later.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "CellDN"
author = "CellDN contributors"
copyright = "2026, CellDN contributors"

try:
    from celldn import __version__ as release
except Exception:
    release = "0.1.0"

version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "titles_only": False,
    "sticky_navigation": True,
}

html_title = "CellDN documentation"
html_static_path: list[str] = []

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

html_context = {
    "display_github": False,
    "github_user": "ShahriyariLab",
    "github_repo": "CellDN",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

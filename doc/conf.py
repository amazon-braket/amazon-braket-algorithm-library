"""Sphinx configuration."""

import datetime
import os
from importlib.metadata import version as pkg_version
from pathlib import Path

from sphinx.application import Sphinx

# Sphinx configuration below.
project = "amazon-braket-algorithm-library"
version = pkg_version(project)
release = version
copyright = f"{datetime.datetime.now().year}, Amazon.com"

extensions = [
    "sphinxcontrib.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
]

source_suffix = ".rst"
master_doc = "index"

autoclass_content = "both"
autodoc_member_order = "bysource"
default_role = "py:obj"

html_theme = "sphinx_rtd_theme"
htmlhelp_basename = f"{project}doc"

napoleon_use_rtype = False

apidoc_module_dir = "../src/braket"
apidoc_output_dir = "_apidoc"
apidoc_excluded_paths = ["../test"]
apidoc_separate_modules = True
apidoc_module_first = True
apidoc_extra_args = ["-f", "--implicit-namespaces", "-H", "API Reference"]

LLMS_TXT_TITLE = "Amazon Braket Algorithm Library"
LLMS_TXT_SUMMARY = (
    "Open source library of pre-built implementations of prominent quantum "
    "algorithms and experimental workloads, provided as ready-to-run example "
    "notebooks for Amazon Braket."
)
LLMS_TXT_BASE_URL = "https://amazon-braket-algorithm-library.readthedocs.io/en/stable/"
LLMS_TXT_SECTIONS: dict[str, tuple[str, ...]] = {
    "Docs": (),
    "API Reference": (f"{apidoc_output_dir}/",),
}


def _llms_txt_section(docname: str) -> str:
    """Return the llms.txt section heading a document belongs under.

    Sections are tried in declaration order, so the first matching prefix wins.
    A document that matches no prefix goes under the first section.
    """
    for heading, prefixes in LLMS_TXT_SECTIONS.items():
        if any(docname.startswith(prefix) for prefix in prefixes):
            return heading
    default_heading, _ = next(iter(LLMS_TXT_SECTIONS.items()))
    return default_heading


def _write_llms_txt(app: Sphinx, exception: Exception | None) -> None:
    """Write llms.txt, a manifest of every built page for LLM discoverability.

    The format follows https://llmstxt.org: an H1 name, a blockquote summary, then
    one file list per H2 section. Pages are grouped so that an agent can tell
    narrative docs and generated API reference apart.
    """
    if exception or app.builder.name != "html":
        return

    # Read the Docs passes the canonical URL to every build automatically, so this
    # is set in any RTD build and the default only applies elsewhere. See
    # https://docs.readthedocs.com/platform/stable/canonical-urls.html#how-to-specify-the-canonical-url
    base_url = os.environ.get("READTHEDOCS_CANONICAL_URL", LLMS_TXT_BASE_URL)
    if base_url and not base_url.endswith("/"):
        base_url += "/"

    env = app.env
    sections: dict[str, list[str]] = {heading: [] for heading in LLMS_TXT_SECTIONS}
    for docname in sorted(env.all_docs):
        url = f"{base_url}{app.builder.get_target_uri(docname)}"
        sections[_llms_txt_section(docname)].append(f"- [{env.titles[docname].astext()}]({url})")

    lines = [f"# {LLMS_TXT_TITLE}", "", f"> {LLMS_TXT_SUMMARY}"]
    for heading in LLMS_TXT_SECTIONS:
        if sections[heading]:
            lines += ["", f"## {heading}", "", *sections[heading]]

    out = Path(app.outdir) / "llms.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"--> Wrote {out.name}")


def setup(app: Sphinx) -> None:
    """Register build hooks."""
    app.connect("build-finished", _write_llms_txt)

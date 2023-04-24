"""Configuration file for the Sphinx documentation builder."""
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import shutil
from datetime import datetime
from importlib.metadata import version as version_func
from pathlib import Path

# import pkg_resources
from sphinx.application import Sphinx

DOCS_DIR = Path(__file__).parent.resolve()

# -- Path setup --------------------------------------------------------------
# We are building and installing the pudl package in order to get access to
# the distribution metadata, including an automatically generated version
# number via pkg_resources.get_distribution() so we need more than just an
# importable path.


# -- Project information -----------------------------------------------------

project = "Dispatch"
copyright = f"{datetime.today().year}, RMI, CC-BY-4.0"  # noqa: A001
author = "RMI"

# The full version, including alpha/beta/rc tags
release = version_func("rmi.dispatch")
version = ".".join(release.split(".")[:2])
html_title = f"{project} {version} documentation"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "sphinx_issues",
]
todo_include_todos = True

# Automatically generate API documentation during the doc build:
autoapi_type = "python"
autoapi_dirs = [
    "../src/dispatch",
]
autoapi_ignore = [
    "*_test.py",
    "*/package_data/*",
    "_*.py",
    "*constants.py",
    "*_version.py",
]
autoapi_python_class_content = "both"
autoapi_options = [
    "members",
    # "undoc-members",
    # "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autodoc_typehints = "description"
# autoapi_keep_files = True

# GitHub repo
issues_github_path = "rmi/dispatch"

# In order to be able to link directly to documentation for other projects,
# we need to define these package to URL mappings:
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numba": ("https://numba.readthedocs.io/en/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "pandera": ("https://pandera.readthedocs.io/en/stable", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "pytest": ("https://docs.pytest.org/en/latest/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "setuptools": ("https://setuptools.pypa.io/en/latest/", None),
    "tox": ("https://tox.wiki/en/latest/", None),
    "etoolbox": ("https://rmi.github.io/etoolbox/", None),
}
"https://plotly.com/python-api-reference/generated/plotly.graph_objects.Figure.html"
"https://plotly.com/python-api-reference/generated/plotly.graph_objs.Figure"
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
master_doc = "index"
html_theme = "furo"
# html_logo = "_static/Small_PNG-RMI_logo_PrimaryUse.PNG"
html_icon = "_static/favicon-16x16.png"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/rmi/dispatch",
            "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
        """,
            "class": "",
        },
    ],
    "navigation_with_keys": True,
    # "logo": {
    #     "image_light": "Small_PNG-RMI_logo_PrimaryUse.PNG",
    #     "image_dark": "Small_PNG-RMI_logo_PrimaryUse_White_Horizontal.PNG",
    # },
    "light_logo": "Small_PNG-RMI_logo_PrimaryUse.PNG",
    "dark_logo": "Small_PNG-RMI_logo_PrimaryUse_White_Horizontal.PNG",
    "source_repository": "https://github.com/rmi/dispatch/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Custom build operations -------------------------------------------------
def cleanup_rsts(app: Sphinx, exception: Exception) -> None:
    """Remove generated RST files when the build is finished."""
    (DOCS_DIR / "path/to/temporary/rst/file.rst").unlink()


def cleanup_csv_dir(app: Sphinx, exception: Exception) -> None:
    """Remove generated CSV files when the build is finished."""
    csv_dir = DOCS_DIR / "path/to/temporary/csv/dir/"
    if csv_dir.exists() and csv_dir.is_dir():
        shutil.rmtree(csv_dir)


def setup(app: Sphinx) -> None:
    """Add custom CSS defined in _static/custom.css."""
    app.add_css_file("custom.css")
    # Examples of custom docs build steps:
    # app.connect("build-finished", cleanup_rsts)
    # app.connect("build-finished", cleanup_csv_dir)

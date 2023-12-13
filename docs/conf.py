import os
import sphinx_rtd_theme
from modulus.sym import __version__ as version
project = 'NVIDIA Modulus Symbolic'
copyright = '2023, NVIDIA Modulus Team'
author = 'NVIDIA Modulus Team'
release = version
exclude_patterns = ['_build', 'external', 'README.md', 'CONTRIBUTING.md',
    'LICENSE.txt', 'tests', '**.ipynb_checkpoints']
autodoc_mock_imports = ['pysdf', 'quadpy', 'functorch']
extensions = ['recommonmark', 'sphinx.ext.mathjax', 'sphinx.ext.todo',
    'sphinx.ext.autosectionlabel', 'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', 'sphinx.ext.viewcode', 'nbsphinx']
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
pdf_documents = [('index', u'rst2pdf', u'Sample rst2pdf doc', u'Your Name')]
napoleon_custom_sections = ['Variable Shape']
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = 'sphinx_rtd_theme'
html_theme_options = {'logo_only': True, 'display_version': True,
    'prev_next_buttons_location': 'bottom', 'style_external_links': False,
    'style_nav_header_background': '#000000', 'collapse_navigation': False,
    'sticky_navigation': False, 'sidebarwidth': 12, 'includehidden': True,
    'titles_only': False}
html_static_path = ['_static']
html_css_files = ['css/nvidia_styles.css']
html_js_files = ['js/pk_scripts.js']
math_number_all = True
todo_include_todos = True
numfig = True
_PREAMBLE = """
\\usepackage{amsmath}
\\usepackage{esint}
\\usepackage{mathtools}
\\usepackage{stmaryrd}
"""
latex_elements = {'preamble': _PREAMBLE}
latex_preamble = [('\\usepackage{amssymb}', '\\usepackage{amsmath}',
    '\\usepackage{amsxtra}', '\\usepackage{bm}', '\\usepackage{esint}',
    '\\usepackage{mathtools}', '\\usepackage{stmaryrd}')]
autosectionlabel_maxdepth = 1

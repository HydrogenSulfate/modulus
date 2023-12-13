from pathlib import Path
from sphinx.application import Sphinx
import logging
logger = logging.getLogger(__name__)


def build_html(src_dir: Path, warning_file: Path=Path('./warnings.txt')):
    """Builds sphinx HTML files

    Parameters
    ----------
    src_dir : Path
        Path to base directory of documentation with conf.py
    warning_file : Path, optional
        File name/ path for logging warnings/errors, by default Path("./warnings.txt")
    """
    conf_dir = src_dir
    build_dir = src_dir / Path('_build')
    doctree_dir = build_dir / Path('doctrees')
    html_dir = build_dir / Path('html')
    builder = 'html'
    try:
        warning_file.unlink()
    except OSError as e:
        print("Couldn't delete old warning file")
    warning = open(warning_file, 'w')
    try:
        build_dir.unlink()
    except OSError as e:
        print("Couldn't delete build directory")
    app = Sphinx(src_dir, conf_dir, html_dir, doctree_dir, builder, warning
        =warning)
    app.build()


if __name__ == '__main__':
    build_html(Path('.'))

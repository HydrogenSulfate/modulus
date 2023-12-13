from pathlib import Path
from sphinx.application import Sphinx
import logging


def check_error_log(warning_file: Path=Path('./warnings.txt'),
    ignore_docstrings: bool=True, ignore_warnings: bool=True):
    """Checks the error log of Sphinx app for errors/warnings

    Parameters
    ----------
    warning_file : Path, optional
        Path to error log files from sphinx, by default Path("./warnings.txt")
    ignore_docstrings : bool, optional
        Ignore docstring related errors, by default True
    ignore_warnings : bool, optional
        Ignore sphinx warnings, by default True

    Raises
    ------
    SystemError
        If error/warning is found which is not ignored
    """
    assert warning_file.is_file(), 'Error log not found'
    errors = []
    with open(warning_file) as file:
        lines = file.read().splitlines()
        i = 0
        print(f'Number of lines in warnings log file: {len(lines)}')
        while i < len(lines):
            line_str = lines[i]
            while not '[39;49;00m' in line_str and i < len(lines):
                i += 1
                line_str = line_str + lines[i]
            i += 1
            if ignore_warnings and 'WARNING' in line_str:
                print(f'Ignoring docstring warning: {line_str}')
                continue
            elif ignore_docstrings and '.py:docstring' in line_str:
                print(f'Ignoring docstring issue: {line_str}')
                continue
            else:
                errors.append(str(line_str))
    if len(errors) > 0:
        print('Errors found when building docs:')
        for error in errors:
            print(error)
        raise SystemError('Sphinx build failed')


if __name__ == '__main__':
    check_error_log(ignore_warnings=False)

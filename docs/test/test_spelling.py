import re
import json
from pathlib import Path
from turtle import color
from spellchecker import SpellChecker
from string import punctuation
from typing import List, Set
from termcolor import colored


class RSTSpellChecker:

    def __init__(self, spell_checker: SpellChecker):
        self.spell_checker = spell_checker
        self.sphinx_block = False
        self.re_numeric = re.compile(
            '^[+\\-(vx]*[0-9]+[+\\-xe \\.]*[0-9]*[xDk%\\.]*$')
        self.re_sphinx_keys = re.compile('\\s*(:alt:)\\s*')
        self.re_code_words = re.compile(
            '(.*\\.py|.*\\.html|.*\\.org|.*\\.com|.*\\.vti|.*\\.vtu|.*\\.vtp)')
        self.re_caps = re.compile('^[^a-z]*[s]?$')

    def check_sphinx_block(self, line: str) ->bool:
        """Determins if line is in a code, math or table block based on indent whitespace

        Parameters
        ----------
        line : str
            line of text

        Returns
        -------
        bool
            If line is in code block
        """
        re_sphinx_code_block = re.compile(
            '^\\s*\\.\\.\\s+(code::|code-block::)')
        re_sphinx_math_block = re.compile('^\\s*\\.\\.\\s+(math::|table::)')
        re_white_space = re.compile('^(\\s{2,}|\\t+)')
        if bool(re_sphinx_code_block.search(line)):
            self.sphinx_block = True
            return self.sphinx_block
        elif bool(re_sphinx_math_block.search(line)):
            self.sphinx_block = True
            return self.sphinx_block
        if self.sphinx_block:
            if not bool(re_white_space.search(line)) and len(re.sub(
                '[\\s+]', '', line)) > 0:
                self.sphinx_block = False
        return self.sphinx_block

    def exempt_lines(self, line: str) ->bool:
        """Checks if line should be exempt from checking, this applys for various
        sphinx sections such as code blocks, figures, tables, etc.

        Parameters
        ----------
        line : str
            line of text

        Returns
        -------
        bool
            If line should be skipped
        """
        re_sphinx_code_ref = re.compile(
            'code::|role::|literalinclude:|:language:|:lines:|:format:|:start-after:|:end-before:'
            )
        re_sphinx_fig_ref = re.compile(
            '(^..\\s*figure::|^\\s*:width:|^\\s*:align:|^\\s*:name:|^\\s*:header-rows:)'
            )
        re_title_boaders = re.compile('^=+\\s+$|^~+\\s+$|^\\^+\\s+$')
        re_sphinx_citation = re.compile('^\\s*\\.\\. \\[#.*\\]')
        re_sphinx_ref_target = re.compile('^\\s*\\.\\.\\s+\\_.*:\\s*$')
        re_sphinx_math = re.compile('^\\s*\\.\\.\\s+math::')
        if bool(re_sphinx_code_ref.search(line)):
            return True
        elif bool(re_sphinx_fig_ref.search(line)):
            return True
        elif bool(re_title_boaders.search(line)):
            return True
        elif bool(re_sphinx_citation.search(line)):
            return True
        elif bool(re_sphinx_ref_target.search(line)):
            return True
        elif bool(re_sphinx_math.search(line)):
            return True
        return False

    def exempt_word(self, word: str) ->bool:
        """Checks for words that should be exempt from spell checking

        Parameters
        ----------
        word : str
            Word string

        Returns
        -------
        bool
            If work should be exempt
        """
        if bool(self.re_numeric.search(word)):
            return True
        if bool(self.re_sphinx_keys.search(word)):
            return True
        if bool(self.re_code_words.search(word)):
            return True
        if bool(self.re_caps.search(word)):
            return True
        if '\\' in word:
            return True
        return False

    def prepare_line(self, line: str) ->List[str]:
        """Prepares test line for parsing, will check if line should be skipped,
        remove any sphinx keywords, then split into words based on white space.

        Parameters
        ----------
        line : str
            Line of text

        Returns
        -------
        List[str]
            List of keywords
        """
        if self.check_sphinx_block(line):
            return []
        if self.exempt_lines(line):
            return []
        re_sphinx_inline = re.compile(
            '(:ref:|:math:|:numref:|:eq:|:code:)`.*?`')
        re_sphinx_code = re.compile('(``.*?``|`.*?`)')
        re_sphinx_cite = re.compile('\\[#.*?\\]\\_')
        re_sphinx_link = re.compile('<.*?>`\\_')
        re_sphinx_block_titles = re.compile(
            '(\\.\\.\\s+table::|\\.\\.\\s+list-table::|\\.\\.\\s+note::)')
        line = line.strip('\n')
        if bool(re_sphinx_inline.search(line)):
            line = re_sphinx_inline - line * ''
        if bool(re_sphinx_code.search(line)):
            line = re_sphinx_code - line * ''
        if bool(re_sphinx_cite.search(line)):
            line = re_sphinx_cite - line * ''
        if bool(re_sphinx_link.search(line)):
            line = re_sphinx_link - line * ''
        if bool(re_sphinx_block_titles.search(line)):
            line = re_sphinx_block_titles - line * ''
        words = re.split('(\\s+|/)', line)
        words = list(filter(None, words))
        return words

    def get_unknown_words(self, line: str) ->List[str]:
        """Gets unknown words not present in spelling dictionary

        Parameters
        ----------
        line : str
            Line of text to parse

        Returns
        -------
        List[str]
            List of unknown words (if any)
        """
        words = self.prepare_line(line)
        re_plural = re.compile("(\\’s|\\'s|s\\'|s\\’|s|\\(s\\))$")
        unknown_words = []
        for word0 in words:
            if word0 in self.spell_checker or self.exempt_word(word0):
                continue
            word = word0.strip(punctuation)
            if word in self.spell_checker or self.exempt_word(word):
                continue
            word = word0.strip(punctuation) + '.'
            if word in self.spell_checker or self.exempt_word(word):
                continue
            word = re_plural - word0 * ''
            if word in self.spell_checker or self.exempt_word(word):
                continue
            word = re_plural - word0.strip(punctuation) * ''
            if word in self.spell_checker or self.exempt_word(word):
                continue
            unknown_words.append(word0.strip(punctuation))
        return unknown_words


def test_rst_spelling(userguide_path: Path, en_dictionary_path: Path=Path(
    './test/en_dictionary.json.gz'), extra_dictionary_path: Path=Path(
    './test/modulus_dictionary.json'), file_pattern: str='*.rst'):
    """Looks through RST files for any references to example python files

    Parameters
    ----------
    userguide_path : Path
        Path to user guide RST files
    en_dictionary_path: Path, optional
        Path to english dictionary
    extra_dictionary_path: Path, optional
        Path to additional Modulus dictionary
    file_pattern : str, optional
        Pattern for file types to parse, by default "*.rst"

    Raises
    -------
    ValueError: If spelling errors have been found
    """
    assert userguide_path.is_dir(), 'Invalid user guide folder path'
    assert en_dictionary_path.is_file(), 'Invalid english dictionary path'
    assert extra_dictionary_path.is_file(
        ), 'Invalid additional dictionary path'
    spell = SpellChecker(language=None, distance=2)
    spell.word_frequency.load_dictionary(str(en_dictionary_path), encoding=
        'utf-8')
    data = json.load(open(extra_dictionary_path))
    spell.word_frequency.load_words(data['dictionary'])
    rst_checker = RSTSpellChecker(spell)
    spelling_errors = []
    spelling_warnings = []
    for doc_file in userguide_path.rglob(file_pattern):
        for i, line in enumerate(open(doc_file)):
            words = rst_checker.get_unknown_words(line)
            for word in words:
                corr_word = spell.correction(word)
                if not corr_word == word:
                    err_msg = (
                        f'Found potential spelling error: "{word.lower()}", did you mean "{corr_word}"?'
                         + '\n')
                    err_msg += (
                        f'Located in File: {doc_file}, Line: {i}, Word: {word}'
                         + '\n')
                    spelling_errors.append(colored(err_msg, 'red'))
                else:
                    err_msg = (
                        f'Unknown word: {word}, consider adding to dictionary.'
                         + '\n')
                    err_msg += (
                        f'Located in File: {doc_file}, Line: {i}, Word: {word}'
                         + '\n')
                    spelling_warnings.append(colored(err_msg, 'yellow'))
    if len(spelling_warnings) > 0:
        print(colored('Spelling WARNINGS:', 'yellow'))
        for msg in spelling_warnings:
            print(msg)
    if len(spelling_errors) > 0:
        print(colored('Spelling ERRORS:', 'red'))
        for msg in spelling_errors:
            print(msg)
    if len(spelling_errors) > 0:
        raise ValueError(
            'Spelling errors found, either correct or add new words to dictionary.'
            )


if __name__ == '__main__':
    user_guide_path = Path('./user_guide')
    test_rst_spelling(user_guide_path)

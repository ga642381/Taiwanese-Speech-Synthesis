""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from utils.text import cmudict
import hparams as hp

# temp, 20201215, since we decided to distinguish ... from .
# we made "..." -> "~"

_punctuation = '~!\'(),.:;? '
_pad = '_'
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_ipa = 'ŋəɛɨʔʰ̩ⁿ'

_sooji = '0123456789'

# We don't need this in Taiwanese synthesis  Kaiwei 2020.12.06
# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + list(_ipa) + list(_sooji)


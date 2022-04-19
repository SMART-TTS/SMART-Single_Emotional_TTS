#-*- coding: utf-8 -*-


'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
#from text import cmudict
from text import HangulUtilsHrim as hangul

#_pad        = '_'
#_eos        = '~'
#_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
#symbols = [_pad, _eos] + list(_characters) + _arpabet

symbols = hangul.hangul_symbols
if __name__ == '__main__':
    print(symbols)

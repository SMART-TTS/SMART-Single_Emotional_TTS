#sh <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh) -*- coding: utf-8 -*-
'''
By Hrim in Human Interface Laboratory
 + BJ Choi adding g2pK module
'''

from g2pk import G2p as g2p
import time
hangul_symbols = u'''␀␃ !,.?ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ'''

dict_cho   = {0:u"ᄀ",  1:u"ᄁ",  2:u"ᄂ",  3:u"ᄃ",  4:u"ᄄ",  5:u"ᄅ",  6:u"ᄆ",  7:u"ᄇ",  8:u"ᄈ",  9:u"ᄉ",
              10:u"ᄊ", 11:u"ᄋ", 12:u"ᄌ", 13:u"ᄎ", 14:u"ᄏ", 15:u"ᄐ", 16:u"ᄎ", 17:u"ᄑ", 18:u"ᄒ"}
dict_jung  = {0:u"ᅡ",  1:u"ᅢ",  2:u"ᅣ",  3:u"ᅤ",  4:u"ᅥ",  5:u"ᅦ",  6:u"ᅧ",  7:u"ᅨ",  8:u"ᅩ",  9:u"ᅪ",
              10:u"ᅫ", 11:u"ᅬ", 12:u"ᅭ", 13:u"ᅮ", 14:u"ᅯ", 15:u"ᅰ", 16:u"ᅱ", 17:u"ᅲ", 18:u"ᅳ", 19:u"ᅴ", 20:u"ᅵ"}
dict_jong  = { 0:u" ",   1:u"ᆨ",  2:u"ᆩ",  3:u"ᆪ",  4:u"ᆫ",  5:u"ᆬ",  6:u"ᆭ",  7:u"ᆮ",  8:u"ᆯ",  9:u"ᆰ",  
               10:u"ᆱ", 11:u"ᆲ", 12:u"ᆳ", 13:u"ᆴ", 14:u"ᆵ", 15:u"ᆶ", 16:u"ᆷ", 17:u"ᆸ", 18:u"ᆹ", 19:u"ᆺ", 
               20:u"ᆻ", 21:u"ᆼ", 22:u"ᆽ", 23:u"ᆾ", 24:u"ᆿ", 25:u"ᇀ", 26:u"ᇁ", 27:u"ᇂ"}

grapheme_to_idx = {char: idx for idx, char in enumerate(hangul_symbols)}
idx_to_grapheme = {idx: char for idx, char in enumerate(hangul_symbols)}


def syllables_to_cjj(syllable_seq): # ##i cjj : Cho/Jung/Jong in Hangul
    cjj = []
    for i in range(len(syllable_seq)):
        if len(syllable_seq[i].encode()) == 3:
            h___ = syllable_seq[i].encode()[0]-224
            _h__ = (syllable_seq[i].encode()[1]-128) // 4
            next_ = (syllable_seq[i].encode()[1]-128) % 4
            __h_ = (next_*64 + syllable_seq[i].encode()[2]-128) // 16
            ___h = (next_*64 + syllable_seq[i].encode()[2]-128) % 16
            hex = h___ * 4096 + _h__ * 256 + __h_ * 16 + ___h

            if hex == 9219: 
                cjj = cjj + [u"␃"] # ##i EOS
                continue
            if hex < 44032:
                continue
            cho  = dict_cho[(hex - 44032) // 588]
            jung = dict_jung[((hex - 44032) % 588) // 28]
            jong  = dict_jong[((hex - 44032) % 588) % 28]
            if jong == u" ": cjj = cjj + [cho, jung]
            else : cjj = cjj + [cho, jung, jong]
        else:
            if syllable_seq[i] not in hangul_symbols: continue
            cjj = cjj + [syllable_seq[i]]
#    print(cjj)
    cjj_id = [grapheme_to_idx[char] for char in cjj]
    return cjj_id

def hangul_to_sequence(hangul_text):
#    start = time.time()
#    G2P = g2p()
#    hangul_text = G2P(hangul_text, descriptive=True, group_vowels=True)
#    print("G2P time", time.time()-start)
#    print(hangul_text)
#    with open('/home/bjchoi/test.txt', 'w') as f:
#        f.write(hangul_text)

    hangul_text = hangul_text + u"␃"  # ␃: EOS

    return syllables_to_cjj(hangul_text)

#m = hangul_to_sequence(hangul_text='이번 역은 낙성대, 낙성대 역입니다, 내리실 문은 없습니다. 헣, 헇! 홇? 숂@ 칵 다 우 다')
#m = hangul_to_sequence(hangul_text='오늘은 5월 17일, 난 32살이야. 이번 역은 2호선 낙성대 역입니다, 내리실')

#k = hangul_to_sequence(hangul_text='ㅇㅣㅂㅓㄴ ㅇㅕㄱㅇㅡㄴ ㄴㅏㄱㅅㅓㅇㄷㅐ ㅇㅕㄱㅇㅣㅂㄴㅣㄷㅏ')
#print(m)
#print(k)

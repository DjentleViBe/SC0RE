"""Decoding results"""
from config import EOS, BOS, BARRE_NOTE, BEND_NOTE_1, BEND_NOTE_7, TREM_BAR_1, TREM_BAR_5, DEAD_NOTE, SLIDE_NOTE_1, SLIDE_NOTE_6

DEMAPPING_BEAT_TYPE = {
    1:   'Base---------------',
    2:   'Triplet------------',
    3:   'Quintuplet---------',
    4:   'Sextuplet----------',
    5:   'Septuplet----------',
    6:   '9_Tuplets----------',
    7:   '11_Tuplets---------',
    8:   'Dotted - Base------',
    9:   'Dotted - Triplet---',
    10:  'Dotted - Quintuplet',
    11:  'Dotted - Sextuplet-',
    12:  'Dotted - Septuplet-',
    13:  'Dotted - 9_Tuplets-',
    14:  'Dotted - 11_Tuplets',
}

def demapping_beat(beat):
    """returns beat type"""
    beat_type = ''
    if beat >= 71:
        beat_type += DEMAPPING_BEAT_TYPE.get(beat - 70)
        return beat_type
    elif 71 > beat >= 57:
        beat_type += DEMAPPING_BEAT_TYPE.get(beat - 56)
        return beat_type
    elif 57 > beat >= 43:
        beat_type += DEMAPPING_BEAT_TYPE.get(beat - 42)
        return beat_type
    elif 43 > beat >= 29:
        beat_type += DEMAPPING_BEAT_TYPE.get(beat - 28)
        return beat_type
    elif 29 > beat >= 15:
        beat_type += DEMAPPING_BEAT_TYPE.get(beat - 14)
        return beat_type
    else:
        beat_type += DEMAPPING_BEAT_TYPE.get(beat + 1)
        return beat_type

def detokenizer_1(dummy):
    """Derive notes from token"""
    note_val = 0
    note_type = 0
    string_num = 0
    palm_mute = False
    if dummy == EOS:
        print(f"-------EOS-------")
        note_val = EOS
    elif dummy == BOS:
        print(f"-------BOS-------")
        note_val = BOS
    elif dummy == BARRE_NOTE:
        print("--------Barred Note--------")
        note_val = BARRE_NOTE
    elif BEND_NOTE_7 >= dummy >= BEND_NOTE_1:
        print("-------Accent_Bend----------")
        note_val = dummy
    elif TREM_BAR_5 >= dummy >= TREM_BAR_1:
        print("-------Accent_Trem----------")
        note_val = dummy
    elif SLIDE_NOTE_6 >= dummy >= SLIDE_NOTE_1:
        print("-------Accent_Slide----------")
        note_val = dummy
    elif dummy == DEAD_NOTE:
        print("-------Dead note----------")
        note_val = dummy
    else:
        palm_mute = False
        beat_type = demapping_beat(dummy // 322)
        note_type = dummy % 322
        if note_type > 161:
            # Palm mute
            note_type -= 161
            palm_mute = True
            string_num = (note_type) // 23 + 1
            note_val = (note_type) % string_num
        else:
            string_num = note_type // 23 + 1
            note_val = note_type % string_num

        print(f"Beat : {beat_type} String : {string_num} Note : {note_val} PalmMute : {palm_mute}")


    return note_val, note_type, string_num, dummy // 322, palm_mute

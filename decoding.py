"""Decoding results"""

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
    if beat >= 69:
        beat_type += DEMAPPING_BEAT_TYPE.get(beat - 68)
        return beat_type
    elif 69 >= beat >= 55:
        beat_type += DEMAPPING_BEAT_TYPE.get(beat - 54)
        return beat_type
    elif 55 >= beat >= 41:
        beat_type += DEMAPPING_BEAT_TYPE.get(beat - 40)
        return beat_type
    elif 41 >= beat >= 29:
        beat_type += DEMAPPING_BEAT_TYPE.get(beat - 28)
        return beat_type
    elif 29 >= beat >= 15:
        beat_type += DEMAPPING_BEAT_TYPE.get(beat - 14)
        return beat_type
    else:
        beat_type += DEMAPPING_BEAT_TYPE.get(beat)
        return beat_type

def detokenizer_1(dummy):
    """Derive notes from token"""
    palm_mute = False
    beat_type = demapping_beat(dummy // 322)
    note_type = dummy % 322
    if note_type > 161:
        # Palm mute
        palm_mute = True
        string_num = (note_type - 161) // 23
        note_val = (note_type - 161) % string_num
    else:
        string_num = note_type // 23
        note_val = note_type % string_num

    print(f"Beat : {beat_type} String : {string_num} Note : {note_val} PalmMute : {palm_mute}")
    return 0

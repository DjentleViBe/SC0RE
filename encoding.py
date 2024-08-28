"""Encoding gpro files"""
# Note
# C0 = 0 in GPro

MAPPING_BEAT = {
    # quarter
    4  :  11,
    5  :  12,
    6  :  9,
    # 8th
    8  :  8,
    9  :  10,
    10 :  6,
    # 16th
    16 :  5,
    17 :  7,
    18 :  4,
    # 32nd
    32 :  2,
    33 :  3,
    34 :  1,
}

def getencodingnotes(note, string, tuning):
    """gives a unique number to note"""
    # total notes = 132
    # String 1 : 0 - 22
    # String 2 : (0 - 22) + (2 - 1) * 22
    encoding = (string - 1) * 22 + note
    return encoding

def getencodingbeats(duration):
    """gives a unique number to beat"""
    # 1 : Triplet-32nd
    # 2 : 32nd
    # 3 : Dotted-32nd
    # 4 : Triplet-16th
    # 5 : 16th
    # 6 : Triplet-8th
    # 7 : Dotted-16th
    # 8 : 8th
    # 9 : Triplet-Quarter
    # 10 : Dotted-8th
    # 11 : Quarter
    # 12 : Dotted-Quarter

    if duration.isDotted:
        duration.value += 1
    if duration.tuplet.enters == 3:
        duration.value += 2
    beatvalue = MAPPING_BEAT.get(duration.value)
    return beatvalue

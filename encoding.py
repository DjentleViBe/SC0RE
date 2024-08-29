"""Encoding gpro files"""
# Note
# C0 = 0 in GPro

MAPPING_BEAT = {
    # pure : base, triplet, quintuplet, sextuplet, septuplet, 9 tuplets, 11-tuplets (1-7)
    # dotted : base, triplet, quintuplet, sextuplet, septuplet, 9 tuplets, 11-tuplets (8-13)
    # full
    1  : 64,
    # half
    2  : 51, 
    # quarter
    4  : 38,
    # 8th
    8  : 25,
    # 16th
    16 : 14,
    # 32nd
    32 : 1  # 1,2,3,4,5,6,7,8,9,10,12,13
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
    # 0 - 132   : # 1
    # 133 - 265 : # 2
    beatvalue = MAPPING_BEAT.get(duration.value)
    if duration.tuplet.enters == 3:
        beatvalue += 1
    if duration.tuplet.enters == 5:
        beatvalue += 2
    if duration.tuplet.enters == 6:
        beatvalue += 3
    if duration.tuplet.enters == 7:
        beatvalue += 4
    if duration.tuplet.enters == 9:
        beatvalue += 5
    if duration.tuplet.enters == 11:
        beatvalue += 6
    if duration.isDotted:
        beatvalue += 8
    return beatvalue

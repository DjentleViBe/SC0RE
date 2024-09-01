"""Encoding gpro files"""
# Note
# C0 = 0 in GPro

MAPPING_BEAT = {
    # pure : base, triplet, quintuplet, sextuplet, septuplet, 9 tuplets, 11-tuplets (1-7)
    # dotted : base, triplet, quintuplet, sextuplet, septuplet, 9 tuplets, 11-tuplets (8-14)
    # full
    1  : 69, # 69 - 82
    # half
    2  : 55, 
    # quarter
    4  : 41,
    # 8th
    8  : 29,
    # 16th
    16 : 15,
    # 32nd
    32 : 1  # 1,2,3,4,5,6,7,8,9,10,11,12,13,14
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

def encoder_1(note, string, duration, n):
    # 7 strings, 23 notes per string, + Palm_Mute
    # 161 * 2 total notes / beat
    # Note formula = n * 23 * String + note, n = 2 if Palm_Mute, else n = 1
    # 
    # String formula = (Note % 23) - 1
    # 82 total beats
    # 161 * 2 * 82 = 26404 unique combinations
    # 1 - 322 =  base note
    # 323 - 644 = triplet
    # .....
    # 
    # Unique combination formula = Note formula + (m * 322), m = mapping_beat number
    note_formula = (note + 1) + (string - 1) * 23 + (7 * 23) * (n - 1)
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
    encoding = note_formula + beatvalue * 322
    return encoding
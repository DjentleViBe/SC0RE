# Note
# C0 = 0 in GPro

MAPPING_NOTE = {
    12: 'C1',
}

def getencoding(note, string, tuning):
    encoding = tuning[string - 1] + note
    return encoding
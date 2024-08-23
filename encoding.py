"""Encoding gpro files"""
# Note
# C0 = 0 in GPro

MAPPING_NOTE = {
    12: 'C1',
}

def getencoding(note, string, tuning):
    """gives a unique number to note"""
    encoding = tuning[string - 1] + note
    return encoding

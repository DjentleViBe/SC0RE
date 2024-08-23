"""Preprocess data required for training"""
import guitarpro

def readgpro(filename):
    """read gpro file"""
    song = guitarpro.parse(filename)
    return song

def getnumstrings(ginfo):
    """get string count"""
    stringcount = len(ginfo.strings)
    print("Number of strings : ", stringcount)
    return stringcount

def gettuning(strings):
    """get tuning config"""
    return strings.value

def guitarinfo(song):
    """accumulate guitar info"""
    tuning = []
    sc_val = getnumstrings(song.tracks[0])
    for i in range(0, sc_val):
        tuning.append(gettuning(song.tracks[0].strings[i]))

    return tuning

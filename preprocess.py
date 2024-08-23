import guitarpro

def readgpro(filename):
    song = guitarpro.parse(filename)
    return song

def getnumstrings(ginfo):
    stringcount = len(ginfo.strings)
    print("Number of strings : ", stringcount)
    return stringcount

def gettuning(strings):
    return strings.value

def guitarinfo(song):
    tuning = []
    sc = getnumstrings(song.tracks[0])
    for i in range(0, sc):
        tuning.append(gettuning(song.tracks[0].strings[i]))
    
    return tuning

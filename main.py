"""Training riffs using ML"""
# import numpy as np
# import guitarpro
from preprocess import readgpro, guitarinfo
from encoding import getencoding

if __name__ == '__main__':
    GPROFOLDER = './gprofiles/'
    song = readgpro(GPROFOLDER + 'Example-1.gp5')
    tuning = guitarinfo(song)
    print("Note Duration")

    for track in song.tracks:
        # Map values to Genaral MIDI.
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    for note in beat.notes:
                        # note.value = encoding.MAPPING_NOTE.get(note.value, note.value)
                        # print(note.value, note.string, note.beat.duration.value)
                        print(getencoding(note.value, note.string, tuning),
                                            note.beat.duration.value)

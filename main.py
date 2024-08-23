"""Training riffs using ML"""
import os
import numpy as np
# import guitarpro
from preprocess import readgpro, guitarinfo
from encoding import getencoding

if __name__ == '__main__':
    training_data = np.zeros((20, 10))
    GPROFOLDER = './gprofiles/'
    j = 0
    k = 0
    for filename in os.listdir(GPROFOLDER):
        file_path = os.path.join(GPROFOLDER, filename)

        # Check if it is a file (not a directory)
        if os.path.isfile(file_path):
            print(f"File: {file_path}")
            song = readgpro(str(file_path))
            tuning = guitarinfo(song)
            print("Note Duration")

            L = 0
            for track in song.tracks:
                # Map values to Genaral MIDI.
                for measure in track.measures:
                    for voice in measure.voices:
                        L = 0
                        for beat in voice.beats:
                            for note in beat.notes:
                                # note.value = encoding.MAPPING_NOTE.get(note.value, note.value)
                                # print(note.value, note.string, note.beat.duration.value)
                                training_data[k][L] = getencoding(note.value, note.string, tuning)
                                L += 1
                    k += 1

print(training_data)

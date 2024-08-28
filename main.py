"""Training riffs using ML"""
import os
import numpy as np
# import guitarpro
import torch
from preprocess import readgpro, guitarinfo
from encoding import getencodingnotes, getencodingbeats
from _encoder.encoder import EncoderAPE

if __name__ == '__main__':
    training_src_notes = np.zeros((5, 10), dtype = 'float32')
    training_data_beats = np.zeros((5, 10), dtype = 'float32')
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
                                training_src_notes[k][L] = getencodingnotes(note.value,
                                                                            note.string,
                                                                            tuning)
                                training_data_beats[k][L] = getencodingbeats(note.beat.duration)
                                L += 1
                    k += 1

    training_src_notes = training_src_notes.reshape(5, 10, 1)
    training_tgt_notes = training_src_notes.copy()
    print("Notes")
    training_src_notes[:,4,:] = 0
    print(training_src_notes)
    #print("Beats")
    #print(training_data_beats)

    
    #training_src_notes[:,4] = 0
    # shifted_arr[:, 1:] = training_src_notes[:, :-1]

    #print("target")
    #print(training_tgt_notes)

    
    ################################ transformers #################################
    ########## Params ##############
    d_model         =   10
    ffn_hidden      =   1024
    max_seq_length  =   7
    num_heads       =   4
    drop_prob       =   0.1
    num_layers      =   1
    learning_rate   =   0.1
    patch           =   2
    stride          =   2
    ################################
    device = torch.device("mps")
    criterion = torch.nn.MSELoss()
    encoder = EncoderAPE(device, d_model, ffn_hidden, max_seq_length, num_heads, drop_prob, num_layers)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr = learning_rate)

    encoder.train()

    num_patch = ((max_seq_length - patch)//stride) + 1
    mask = torch.full([num_patch, num_patch], float('-inf'))
    mask = torch.triu(mask, diagonal = 1).to(device)

    with torch.set_grad_enabled(True):
        src = torch.from_numpy(training_src_notes).to(device)
        target = torch.from_numpy(training_tgt_notes).to(device)

        prediction_val = encoder(src).to(device)
        loss = criterion(target, prediction_val)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()







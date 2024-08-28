"""Training riffs using ML"""
import os
import numpy as np
# import guitarpro
import torch
from torch import nn
from preprocess import readgpro, guitarinfo, get_positional_encoding
from encoding import getencodingnotes, getencodingbeats
from _encoder.encoder import EncoderAPE

if __name__ == '__main__':
    training_src_notes = np.zeros((5, 10), dtype = 'int32')
    training_data_beats = np.zeros((5, 10), dtype = 'int32')
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

    # training_src_notes = training_src_notes.reshape(5, 10, 1)
    training_tgt_notes = training_src_notes.copy()
    print("Notes")
    training_src_notes[:,4] = 0
    print(training_src_notes)
    #print("Beats")
    #print(training_data_beats)


    #training_src_notes[:,4] = 0
    # shifted_arr[:, 1:] = training_src_notes[:, :-1]

    #print("target")
    #print(training_tgt_notes)


    ################################ transformers #################################
    ########## Params ##############
    D_MODEL         =   512
    VOCAL_SIZE      =   132
    FFN_HIDDEN      =   1024
    MAX_SEQ_LENGTH  =   10
    NUM_HEADS       =   4
    DROP_PROB       =   0.1
    NUM_LAYERS      =   1
    LEARNING_RATE   =   0.1
    PATCH           =   2
    STRIDE          =   2
    ################################
    device = torch.device("mps")
    criterion = torch.nn.MSELoss()
    encoder = EncoderAPE(device, D_MODEL, FFN_HIDDEN, MAX_SEQ_LENGTH, NUM_HEADS,
                         DROP_PROB, NUM_LAYERS)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr = LEARNING_RATE)

    encoder.train()

    NUM_PATCH = ((MAX_SEQ_LENGTH - PATCH)//STRIDE) + 1
    mask = torch.full([NUM_PATCH, NUM_PATCH], float('-inf'))
    mask = torch.triu(mask, diagonal = 1).to(device)

    with torch.set_grad_enabled(True):
        embedding_layer = nn.Embedding(num_embeddings = VOCAL_SIZE,
                                       embedding_dim = D_MODEL).to(device)
        token_ids = torch.tensor(training_src_notes).to(device)

        embeddings = embedding_layer(token_ids)
        pos_enc = get_positional_encoding(10, D_MODEL).to(device)

        input_embeddings = embeddings + pos_enc

        # src = torch.from_numpy(training_src_notes).to(device)
        # target = torch.from_numpy(training_tgt_notes).to(device)

        prediction_val = encoder(input_embeddings).to(device)
        loss = criterion(target, prediction_val)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

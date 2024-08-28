"""Training riffs using ML"""
import os
import numpy as np
# import guitarpro
import torch
from torch import nn
import torch.nn.functional as F
from preprocess import readgpro, guitarinfo, get_positional_encoding
from encoding import getencodingnotes, getencodingbeats
# from _encoder.encoder import EncoderAPE
from _decoder.decoder import DecoderAPE

if __name__ == '__main__':

    ################################ transformers #################################
    ########## Params ##############
    EPOCHS          =   100
    D_MODEL         =   512
    VOCAB_SIZE      =   132
    FFN_HIDDEN      =   1024
    MAX_SEQ_LENGTH  =   5
    NUM_HEADS       =   4
    DROP_PROB       =   0.1
    NUM_LAYERS      =   1
    LEARNING_RATE   =   0.1
    PATCH           =   1
    STRIDE          =   1
    ################################

    training_src_notes = np.zeros((5, MAX_SEQ_LENGTH), dtype = 'int32')
    training_data_beats = np.zeros((5, MAX_SEQ_LENGTH), dtype = 'int32')
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
    # print("Notes")
    training_src_notes[:,4] = 0
    print("Source")
    print(training_src_notes)
    print("Target")
    print(training_tgt_notes)
    #print("Beats")
    #print(training_data_beats)


    #training_src_notes[:,4] = 0
    # shifted_arr[:, 1:] = training_src_notes[:, :-1]

    #print("target")
    #print(training_tgt_notes)

    iteration = 0
    device = torch.device("mps")
    criterion = torch.nn.MSELoss()
    # encoder = EncoderAPE(device, D_MODEL, FFN_HIDDEN, MAX_SEQ_LENGTH, NUM_HEADS,
    #                     DROP_PROB, NUM_LAYERS)
    decoder = DecoderAPE(device, D_MODEL, VOCAB_SIZE, FFN_HIDDEN, MAX_SEQ_LENGTH, NUM_HEADS,
                         DROP_PROB, NUM_LAYERS).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr = LEARNING_RATE)

    # encoder.train()

    NUM_PATCH = ((MAX_SEQ_LENGTH - PATCH)//STRIDE) + 1
    mask = torch.full([NUM_PATCH, NUM_PATCH], float('-inf'))
    mask = torch.triu(mask, diagonal = 1).to(device)
    loss_fn = nn.CrossEntropyLoss()

    embedding_layer = nn.Embedding(num_embeddings = VOCAB_SIZE,
                                    embedding_dim = D_MODEL).to(device)
    token_ids = torch.tensor(training_src_notes).to(device)


    pos_enc = get_positional_encoding(MAX_SEQ_LENGTH, D_MODEL).to(device)
    # src = torch.from_numpy(training_src_notes).to(device)
    target = torch.from_numpy(training_tgt_notes).to(device)
    # prediction_val = encoder(input_embeddings).to(device)

    while iteration <= EPOCHS:
        decoder.train()

        optimizer.zero_grad()
        embeddings = embedding_layer(token_ids)
        input_embeddings = embeddings + pos_enc

        logits = decoder(target, input_embeddings, mask)
        # probabilities = F.softmax(prediction_val, dim=-1)

        logits = logits.view(-1, VOCAB_SIZE)  # Flatten logits to (batch_size * seq_length, d_vocab)
        target = target.view(-1)

        loss = loss_fn(logits, target)
        # loss = criterion(target, prediction_val)

        print(iteration + 1, loss.item())
        loss.backward()
        optimizer.step()

        iteration += 1

        if loss.item() < 0.001:
            break

    decoder.eval()
    dummy_in = torch.tensor(np.array([[89, 0, 0, 0, 0]], dtype = 'int32')).to(device)
    embeddings = embedding_layer(dummy_in)
    pos_enc_eval = get_positional_encoding(MAX_SEQ_LENGTH, D_MODEL).to(device)
    output_eval = decoder(target, embeddings + pos_enc_eval, mask)

    next_token_logits = output_eval[:, -1, :]
    probabilities = F.softmax(next_token_logits, dim=-1)

    # Select the next token (using greedy search here)
    next_token = torch.argmax(probabilities, dim=-1).unsqueeze(0)

    generated_sequence = torch.cat([dummy_in, next_token], dim=1)
    print(generated_sequence)
    print("Finished")


"""Training riffs using ML"""
import os
import sys
import numpy as np
# import guitarpro
import torch
from torch import nn
from preprocess import readgpro, guitarinfo, get_positional_encoding, create_dir
from postprocess import plot, decoder_inference
from encoding import tokenizer_1
from decoding import detokenizer_1
# from _encoder.encoder import EncoderAPE
from _decoder.decoder import DecoderAPE
np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':
    create_dir('./RESULTS/')
    ################################ transformers #################################
    MODE            =   0  # 0: train, # 1 : eval, # 2 : both
    BACKUP          =   "dec_only_notes_5"
    START_ID        =   9592
    ########## Params ##############
    EPOCHS          =   1000
    VOCAB_SIZE      =   26412
    FFN_HIDDEN      =   1024
    MAX_SEQ_LENGTH  =   25
    NUM_HEADS       =   2
    DROP_PROB       =   0.3
    NUM_LAYERS      =   2
    D_MODEL         =   60
    LEARNING_RATE   =   0.01
    PATCH           =   1
    STRIDE          =   1
    TRAINING        =   ["CB"]
    BATCH           =   424

    EOS             =   26405
    BOS             =   26406
    BARRE_NOTE      =   26407
    if os.name == 'posix':
        DEVICE_TYPE     =   "mps"
    else:
        DEVICE_TYPE     =   "cuda"
    ################################
    NUM_PATCH = ((MAX_SEQ_LENGTH - PATCH)//STRIDE) + 1
    device = torch.device(DEVICE_TYPE)
    decoder = DecoderAPE(device, D_MODEL, VOCAB_SIZE, FFN_HIDDEN, MAX_SEQ_LENGTH, NUM_HEADS,
                            DROP_PROB, NUM_LAYERS).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr = LEARNING_RATE)
    embedding_layer = nn.Embedding(num_embeddings = VOCAB_SIZE,
                                        embedding_dim = D_MODEL).to(device)
    pos_enc = get_positional_encoding(MAX_SEQ_LENGTH, D_MODEL).to(device)
    mask = torch.full([NUM_PATCH, NUM_PATCH], float('-inf'))
    mask = torch.triu(mask, diagonal = 1).to(device)

    if MODE in (0, 2):
        training_src_encoder_1 = np.zeros((BATCH * MAX_SEQ_LENGTH), dtype = 'int32')

        GPROFOLDER = './gprofiles/'
        j = 0
        k = 0
        L = 0
        for f in TRAINING:
            for filename in os.listdir(GPROFOLDER + f):
                file_path = os.path.join(GPROFOLDER + f, filename)

                # Check if it is a file (not a directory)
                if os.path.isfile(file_path):
                    print(f"File: {file_path}")
                    song = readgpro(str(file_path))
                    tuning = guitarinfo(song)

                    for track in song.tracks:
                        # Map values to Genaral MIDI.
                        for measure in track.measures:
                            #for voice in measure.voices:
                            #print(f"Voice {voice.index}:")
                            # L = 0

                            training_src_encoder_1[L] = BOS
                            L += 1
                            for beat in measure.voices[0].beats:
                                for note_index, note in enumerate(beat.notes):
                                    training_src_encoder_1[L] = tokenizer_1(note.value,
                                                                        note.string,
                                                                        note.beat.duration,
                                                                        note.effect.palmMute + 1)
                                    L += 1
                                    if note_index != 0:
                                        training_src_encoder_1[L] = BARRE_NOTE
                                        L += 1
                            training_src_encoder_1[L] = EOS
                            L += 1
                        # L += 1
                        k += 1

        training_src_encoder_1 = training_src_encoder_1.reshape(BATCH, MAX_SEQ_LENGTH)
        training_tgt_notes = training_src_encoder_1.copy().astype(np.int64)
        print("Source")
        print(f"{training_src_encoder_1}")

        ITERATION = 0
        criterion = torch.nn.MSELoss()
        lossplot = []
        loss_fn = nn.CrossEntropyLoss()

        token_ids = torch.tensor(training_src_encoder_1).to(device)
        target = torch.from_numpy(training_tgt_notes).to(device)

        while ITERATION <= EPOCHS:
            decoder.train()

            optimizer.zero_grad()
            embeddings = embedding_layer(token_ids)
            input_embeddings = embeddings + pos_enc

            logits = decoder(input_embeddings, mask)
            # Flatten logits to (batch_size * seq_length, d_vocab)
            logits = logits.view(-1, VOCAB_SIZE)

            target = target.view(-1)

            loss = loss_fn(logits, target)

            print(f"{ITERATION + 1} : {loss.item()}")
            loss.backward()
            optimizer.step()

            ITERATION += 1
            lossplot.append(loss.item())

            if loss.item() < 0.001:
                break

        checkpoint = {
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': EPOCHS,  # Optionally save the epoch number
        'loss': loss     # Optionally save the loss value
        }
        torch.save(checkpoint, './RESULTS/'+ BACKUP +'.pth')
        plot(lossplot, './RESULTS/' + BACKUP + '.png')

    if MODE in (1, 2):
        checkpoint = torch.load('./RESULTS/'+ BACKUP +'.pth')
        decoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loss : ', {checkpoint['loss'].item()}")
        print(f"Epochs : ', {checkpoint['epoch']}")
        decoder.eval()
        dummy_np = np.full((1, MAX_SEQ_LENGTH), EOS, dtype = 'int32')
        dummy_np[0, 0] = BOS
        dummy_np[0, 1] = START_ID
        dummy_in = torch.tensor(dummy_np).to(device)
        dummy_in = decoder_inference(decoder, dummy_in, embedding_layer, pos_enc, mask,
                                     MAX_SEQ_LENGTH).cpu().numpy()
        for dummy in dummy_in[0]:
            detokenizer_1(dummy)

    print("Finished")

"""Training riffs using ML"""
import os
import sys
import shutil
import numpy as np
import torch
from torch import nn
from preprocess import readgpro, guitarinfo, get_positional_encoding, create_dir
from postprocess import plot, decoder_inference, makegpro
from encoding import tokenizer_1
from decoding import detokenizer_1
from _decoder.decoder import DecoderAPE
import config as cfg
np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':
    create_dir('./RESULTS/')

    if os.name == 'posix':
        DEVICE_TYPE     =   "mps"
    else:
        DEVICE_TYPE     =   "cuda"
    ################################
    NUM_PATCH = ((cfg.MAX_SEQ_LENGTH - cfg.PATCH)//cfg.STRIDE) + 1
    device = torch.device(DEVICE_TYPE)
    decoder = DecoderAPE(device, cfg.D_MODEL, cfg.VOCAB_SIZE, cfg.FFN_HIDDEN, cfg.MAX_SEQ_LENGTH,
                         cfg.NUM_HEADS, cfg.DROP_PROB, cfg.NUM_LAYERS).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr = cfg.LEARNING_RATE)
    embedding_layer = nn.Embedding(num_embeddings = cfg.VOCAB_SIZE,
                                        embedding_dim = cfg.D_MODEL).to(device)
    pos_enc = get_positional_encoding(cfg.MAX_SEQ_LENGTH, cfg.D_MODEL).to(device)
    mask = torch.full([NUM_PATCH, NUM_PATCH], float('-inf'))
    mask = torch.triu(mask, diagonal = 1).to(device)

    if cfg.MODE in (0, 2):
        shutil.copy("./config.py", "./RESULTS/" + cfg.BACKUP + ".py")
        training_src_encoder_1 = np.zeros((cfg.BATCH * cfg.MAX_SEQ_LENGTH), dtype = 'int32')

        GPROFOLDER = './gprofiles/'
        L = 0
        for f in cfg.TRAINING:
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
                            training_src_encoder_1[L] = cfg.BOS
                            L += 1
                            for beat in measure.voices[0].beats:
                                for note_index, note in enumerate(beat.notes):
                                    training_src_encoder_1[L] = tokenizer_1(note.value,
                                                                        note.string,
                                                                        note.beat.duration,
                                                                        note.effect.palmMute + 1)
                                    L += 1
                                    if note_index != 0:
                                        training_src_encoder_1[L] = cfg.BARRE_NOTE
                                        L += 1

                                    if note.effect.isBend > 0:
                                        if note.effect.bend.type.value == 1:
                                            training_src_encoder_1[L] = cfg.BEND_NOTE_1
                                        elif note.effect.bend.type.value == 2:
                                            training_src_encoder_1[L] = cfg.BEND_NOTE_2
                                        elif note.effect.bend.type.value == 3:
                                            training_src_encoder_1[L] = cfg.BEND_NOTE_3
                                        elif note.effect.bend.type.value == 4:
                                            training_src_encoder_1[L] = cfg.BEND_NOTE_4
                                        elif note.effect.bend.type.value == 5:
                                            training_src_encoder_1[L] = cfg.BEND_NOTE_5
                                        elif note.effect.bend.type.value == 6:
                                            training_src_encoder_1[L] = cfg.BEND_NOTE_6
                                        elif note.effect.bend.type.value == 7:
                                            training_src_encoder_1[L] = cfg.BEND_NOTE_7
                                        L += 1

                                    if beat.effect.isTremoloBar is True:
                                        if beat.effect.tremoloBar.type.value == 1:
                                            training_src_encoder_1[L] = cfg.TREM_BAR_1
                                        elif beat.effect.tremoloBar.type.value == 2:
                                            training_src_encoder_1[L] = cfg.TREM_BAR_2
                                        elif beat.effect.tremoloBar.type.value == 3:
                                            training_src_encoder_1[L] = cfg.TREM_BAR_3
                                        elif beat.effect.tremoloBar.type.value == 4:
                                            training_src_encoder_1[L] = cfg.TREM_BAR_4
                                        elif beat.effect.tremoloBar.type.value == 5:
                                            training_src_encoder_1[L] = cfg.TREM_BAR_5
                                        L += 1

                            training_src_encoder_1[L] = cfg.EOS
                            L += 1

        training_src_encoder_1 = training_src_encoder_1.reshape(cfg.BATCH, cfg.MAX_SEQ_LENGTH)
        training_tgt_notes = training_src_encoder_1.copy().astype(np.int64)
        print("Source")
        print(f"{training_src_encoder_1}")

        ITERATION = 0
        criterion = torch.nn.MSELoss()
        lossplot = []
        loss_fn = nn.CrossEntropyLoss()

        token_ids = torch.tensor(training_src_encoder_1).to(device)
        target = torch.from_numpy(training_tgt_notes).to(device)

        while ITERATION <= cfg.EPOCHS:
            decoder.train()

            optimizer.zero_grad()
            embeddings = embedding_layer(token_ids)
            input_embeddings = embeddings + pos_enc

            logits = decoder(input_embeddings, mask)
            # Flatten logits to (batch_size * seq_length, d_vocab)
            logits = logits.view(-1, cfg.VOCAB_SIZE)
            scaled_logits = logits / cfg.TEMPERATURE

            target = target.view(-1)

            loss = loss_fn(scaled_logits, target)

            print(f"{ITERATION + 1} : {loss.item()}")
            loss.backward()
            optimizer.step()

            ITERATION += 1
            lossplot.append(loss.item())

            if loss.item() < cfg.CONVERGENCE:
                break

        checkpoint = {
        'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': cfg.EPOCHS,  # Optionally save the epoch number
        'loss': loss     # Optionally save the loss value
        }
        torch.save(checkpoint, './RESULTS/'+ cfg.BACKUP +'.pth')
        plot(lossplot, './RESULTS/' + cfg.BACKUP + '.png')

    if cfg.MODE in (1, 2):
        checkpoint = torch.load('./RESULTS/'+ cfg.BACKUP +'.pth')
        decoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loss : ', {checkpoint['loss'].item()}")
        print(f"Epochs : ', {checkpoint['epoch']}")
        decoder.eval()
        dummy_np = np.full((1, cfg.MAX_SEQ_LENGTH), cfg.EOS, dtype = 'int32')
        dummy_np[0, 0] = cfg.BOS
        dummy_np[0, 1] = cfg.START_ID
        dummy_in = torch.tensor(dummy_np).to(device)
        dummy_in = decoder_inference(decoder, dummy_in, embedding_layer, pos_enc, mask,
                                     cfg.MAX_SEQ_LENGTH).cpu().numpy()
        noteval = []
        notetypeval = []
        stringnum = []
        beatval = []
        palmval = []
        for ind, dummy in enumerate(dummy_in[0]):
            print(f"{ind + 1:02}", end=' ')
            note, notetype, string, beat, palm = detokenizer_1(dummy)
            noteval.append(note)
            notetypeval.append(notetype)
            stringnum.append(string)
            beatval.append(beat)
            palmval.append(palm)
        makegpro(cfg.SAVE, noteval, notetypeval, stringnum, beatval, palmval)
    print("Finished")

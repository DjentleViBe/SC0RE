"""Post processing"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import guitarpro as gp
from config import EOS, BOS, BARRE_NOTE, MEASURE

DEMAPPING_BEAT_DETYPE = {
    'Base---------------' : 1,
    'Triplet------------' : 2,
    'Quintuplet---------' : 3,
    'Sextuplet----------' : 4,
    'Septuplet----------' : 5,
    '9_Tuplets----------' : 6,
    '11_Tuplets---------' : 7,
    'Dotted - Base------' : 8,
    'Dotted - Triplet---' : 9,
    'Dotted - Quintuplet' : 10,
    'Dotted - Sextuplet-' : 11,
    'Dotted - Septuplet-' : 12,
    'Dotted - 9_Tuplets-' : 13,
    'Dotted - 11_Tuplets' : 14,
}

def plot(lossplot, filename):
    "Plots data and saves figure"
    plt.plot(lossplot)
    plt.suptitle('SupervisedRiffGen')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig(filename, dpi = 200)

    return 0

def decoder_inference(decoder, dummy_in, embedding_layer, pos_enc, mask, seq_lim):
    """Transformer Decoder"""
    for e_val in range (2, seq_lim):
        embeddings = embedding_layer(dummy_in)
        output_eval = decoder(embeddings + pos_enc, mask)

        next_token_logits = output_eval[:, -1, :]
        probabilities = F.softmax(next_token_logits, dim=-1)

        # Select the next token (using greedy search here)
        next_token = torch.argmax(probabilities, dim=-1).unsqueeze(0)

        # generated_sequence = dummy_in
        dummy_in[0][e_val] = next_token
    print(dummy_in)
    return dummy_in

def getnotetype(notetype):
    """Return note type"""
    if 1 <= notetype <= 14:
        return 32, notetype
    elif 15 <= notetype <= 28:
        return 16, notetype - 14
    elif 29 <= notetype <= 40:
        return 8, notetype - 28
    elif 41 <= notetype <= 54:
        return 4, notetype - 40
    elif 54 <= notetype < 68:
        return 2, notetype - 55
    else:
        return 1, notetype - 68

def makegpro(filename, noteval, notetypeval, stringnum, beatval, palmval):
    """Generate gpro file"""
    
    # Create a new Guitar Pro song
    song = gp.models.Song()

    # Set the song's information
    song.title = filename
    song.artist = "DjentleViBe"
    song.tempo = 120  # Set the tempo
    song.tracks[0].name = "Guitar"
    song.tracks[0].channel.instrument = 30
    song.tracks[0].measures[0].timeSignature.numerator = MEASURE
    song.tracks[0].strings[0].value = 58
    song.tracks[0].strings[1].value = 53
    song.tracks[0].strings[2].value = 49
    song.tracks[0].strings[3].value = 44
    song.tracks[0].strings[4].value = 39
    song.tracks[0].strings[5].value = 32
    

    voice = song.tracks[0].measures[0].voices[0]

    k = 0
    l = 0
    beat_collect = []
    note_collect = []
    for n, note in enumerate(noteval):
        if note == EOS:
            continue
        elif note == BOS:
            continue
        elif note == BARRE_NOTE:
            l -= 1
            print(BARRE_NOTE)
        else:
            beat_collect.append(gp.Beat(voice=voice))
            voice.beats.append(beat_collect[k])
            note_collect.append(gp.Note(beat = beat_collect[k]))
            note_collect[l].value = note
            note_collect[l].effect.palmMute = palmval[n]
            note_collect[l].string = min(stringnum[n], 6)
            
            # beat_val = note_collect[k].beat
            note_collect[l].beat.duration.value, b_val = getnotetype(beatval[n])
            # print(n, 
            #      note_collect[k].beat.duration.value, 
            #      b_val, note, 
            #      str(note_collect[k].string), 
            #      palmval[n])
            
            if b_val >= 8:
                note_collect[k].beat.duration.isDotted = True
            if b_val == 2 or b_val == 9:
                note_collect[k].beat.duration.tuplet.enters = 3
                note_collect[k].beat.duration.tuplet.times = 2
            if b_val == 3 or b_val == 10:
                note_collect[k].beat.duration.tuplet.enters = 5
                note_collect[k].beat.duration.tuplet.times = 4
            if b_val == 4 or b_val == 11:
                note_collect[k].beat.duration.tuplet.enters = 6
                note_collect[k].beat.duration.tuplet.times = 4
            if b_val == 5 or b_val == 12:
                note_collect[k].beat.duration.tuplet.enters = 7
                note_collect[k].beat.duration.tuplet.times = 4
            if b_val == 6 or b_val == 13:
                note_collect[k].beat.duration.tuplet.enters = 9
                note_collect[k].beat.duration.tuplet.times = 8
            if b_val == 7 or b_val == 14:
                note_collect[k].beat.duration.tuplet.enters = 11
                note_collect[k].beat.duration.tuplet.times = 8
            

            beat_collect[k].notes.append(note_collect[l])
            k += 1
            l += 1

    # Save the song to a Guitar Pro file
    with open("./RESULTS/" + filename + ".gp5", 'wb') as file:
        gp.write(song, file)

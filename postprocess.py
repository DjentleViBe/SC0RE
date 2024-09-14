"""Post processing"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import guitarpro as gp
from config import (EOS, BOS, BARRE_NOTE, MEASURE, BEND_NOTE_1, BEND_NOTE_2, BEND_NOTE_3,
BEND_NOTE_4, BEND_NOTE_5, BEND_NOTE_6, BEND_NOTE_7, TREM_BAR_1, TREM_BAR_2, TREM_BAR_3,
TREM_BAR_4, TREM_BAR_5)

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
    elif 55 <= notetype <= 68:
        return 2, notetype - 54
    elif 69 <= notetype <= 82:
        return 1, notetype - 68

def makegpro(titlename, noteval, stringnum, beatval, palmval):
    """Generate gpro file"""
    # read bend and tremolo templates
    song_trem_1 = gp.parse('./gprofiles/trem_1.gp5')
    song_trem_2 = gp.parse('./gprofiles/trem_2.gp5')
    song_trem_4 = gp.parse('./gprofiles/trem_4.gp5')
    song_trem_5 = gp.parse('./gprofiles/trem_5.gp5')
    trem1_beat = song_trem_1.tracks[0].measures[0].voices[0].beats[0]
    trem2_beat = song_trem_2.tracks[0].measures[0].voices[0].beats[0]
    trem4_beat = song_trem_4.tracks[0].measures[0].voices[0].beats[0]
    trem5_beat = song_trem_5.tracks[0].measures[0].voices[0].beats[0]

    # Create a new Guitar Pro song
    song = gp.models.Song()

    # Set the song's information
    song.title = titlename
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

    k_val = 0
    l_val = 0
    beat_collect = []
    note_collect = []
    for n, note in enumerate(noteval):

        if note == EOS:
            # print("-----EOS-----")
            continue
        elif note == BOS:
            # print("-----BOS-----")
            continue
        elif note == BARRE_NOTE:
            l_val -= 1
            # print("-----Barred Note-----")
        elif TREM_BAR_1 <= note <= TREM_BAR_5:
            beat_collect[l_val - 1].effect.isBend = True
            
            if note == TREM_BAR_1:
                trem1_beat.notes[0].value = note_collect[l_val - 1].value
                trem1_beat.notes[0].string = note_collect[l_val - 1].string
                trem1_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = trem1_beat
            elif note == TREM_BAR_2:
                trem2_beat.notes[0].value = note_collect[l_val - 1].value
                trem2_beat.notes[0].string = note_collect[l_val - 1].string
                trem2_beat.notes[0].beat.duration.value = beat_collect[l_val - 1].duration.value
                voice.beats[l_val - 1] = trem2_beat
            elif note == TREM_BAR_3:
                beat_collect[l_val - 1].effect.tremoloBar.type = 3
            elif note == TREM_BAR_4:
                trem4_beat.notes[0].value = note_collect[l_val - 1].value
                trem4_beat.notes[0].string = note_collect[l_val - 1].string
                trem4_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = trem4_beat
            elif note == TREM_BAR_5:
                trem5_beat.notes[0].value = note_collect[l_val - 1].value
                trem5_beat.notes[0].string = note_collect[l_val - 1].string
                trem5_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = trem5_beat
            continue
            
        elif BEND_NOTE_1 <= note <= BEND_NOTE_7:
            """
            note_collect[l_val - 1].effect.isBend = True
            if note == BEND_NOTE_1:
                note_collect[l_val - 1].effect.bend.type.value = 1
            elif note == BEND_NOTE_2:
                note_collect[l_val - 1].effect.bend.type.value = 2
            elif note == BEND_NOTE_3:
                note_collect[l_val - 1].effect.bend.type.value = 3
            elif note == BEND_NOTE_4:
                note_collect[l_val - 1].effect.bend.type.value = 4
            elif note == BEND_NOTE_5:
                note_collect[l_val - 1].effect.bend.type.value = 5
            elif note == BEND_NOTE_6:
                note_collect[l_val - 1].effect.bend.type.value = 6
            elif note == BEND_NOTE_7:
                note_collect[l_val - 1].effect.bend.type.value = 7
            continue"""
        else:
            beat_collect.append(gp.Beat(voice=voice))
            voice.beats.append(beat_collect[k_val])
            note_collect.append(gp.Note(beat = beat_collect[k_val]))
            note_collect[l_val].value = note
            note_collect[l_val].effect.palmMute = palmval[n]
            note_collect[l_val].string = min(stringnum[n], 6)

            note_collect[l_val].beat.duration.value, b_val = getnotetype(beatval[n])

            if b_val >= 8:
                note_collect[k_val].beat.duration.isDotted = True
            if b_val in (2, 9):
                note_collect[k_val].beat.duration.tuplet.enters = 3
                note_collect[k_val].beat.duration.tuplet.times = 2
            if b_val in (3, 10):
                note_collect[k_val].beat.duration.tuplet.enters = 5
                note_collect[k_val].beat.duration.tuplet.times = 4
            if b_val in (4, 11):
                note_collect[k_val].beat.duration.tuplet.enters = 6
                note_collect[k_val].beat.duration.tuplet.times = 4
            if b_val in (5, 12):
                note_collect[k_val].beat.duration.tuplet.enters = 7
                note_collect[k_val].beat.duration.tuplet.times = 4
            if b_val in (6, 13):
                note_collect[k_val].beat.duration.tuplet.enters = 9
                note_collect[k_val].beat.duration.tuplet.times = 8
            if b_val in (7, 14):
                note_collect[k_val].beat.duration.tuplet.enters = 11
                note_collect[k_val].beat.duration.tuplet.times = 8

            beat_collect[k_val].notes.append(note_collect[l_val])
            k_val += 1
            l_val += 1
    
    return song

def writegpro(filename, song):
    """write gpro file to disk"""
    # Save the song to a Guitar Pro file
    with open("./RESULTS/" + filename + ".gp5", 'wb') as file:
        gp.write(song, file)

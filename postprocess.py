"""Post processing"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import guitarpro as gp
import math
from config import (BACKUP, MAX_SEQ_LENGTH, EOS, BOS, BARRE_NOTE, MEASURE, BEND_NOTE_1, BEND_NOTE_2, BEND_NOTE_3,
BEND_NOTE_4, BEND_NOTE_5, BEND_NOTE_6, BEND_NOTE_7, TREM_BAR_1, TREM_BAR_2, TREM_BAR_3,
TREM_BAR_4, TREM_BAR_5, DEAD_NOTE, SLIDE_NOTE_1, SLIDE_NOTE_2, SLIDE_NOTE_3, SLIDE_NOTE_4, SLIDE_NOTE_5, SLIDE_NOTE_6,
HAMMER, VIBRATO, HARMONIC_1, 
TEMPERATURE, TEST_CRITERIA, PREDICTION_CRITERIA)

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

def decoder_greedy_search(decoder, dummy_in, embedding_layer, pos_enc, mask):
    embeddings = embedding_layer(dummy_in)
    output_eval = decoder(embeddings + pos_enc, mask)

    next_token_logits = output_eval[:, -1, :]
    scaled_logits = next_token_logits / TEMPERATURE
    
    probabilities = F.softmax(scaled_logits, dim=-1)
    

    # Select the next token (using greedy search here)
    if PREDICTION_CRITERIA == 1:
        next_token = torch.argmax(probabilities, dim=-1).unsqueeze(0)
    elif PREDICTION_CRITERIA == 3:
        next_token = torch.multinomial(probabilities, num_samples=1).unsqueeze(0)

    return next_token

def decoder_inference(decoder, dummy_in, embedding_layer, pos_enc, mask, seq_lim):
    """Transformer Decoder"""
    if TEST_CRITERIA != 4:
        for e_val in range (2, seq_lim):
            next_token = decoder_greedy_search(decoder, dummy_in, embedding_layer, pos_enc, mask)
            # generated_sequence = dummy_in
            dummy_in[0][e_val] = next_token
    else:
        e_val = 2
        trial = 0
        temperature_var = TEMPERATURE
        while e_val < seq_lim:
            next_token = decoder_greedy_search(decoder, dummy_in, embedding_layer, pos_enc, mask)

            if e_val != 2:
                if next_token > BOS and dummy_in[0][e_val - 1] < BOS:
                    dummy_in[0][e_val] = next_token
                    e_val +=1
                elif next_token < BOS:
                    dummy_in[0][e_val] = next_token
                    e_val +=1
                else:
                    temperature_var /= 2
                    print("Trial -->", e_val, trial, next_token.cpu()[0][0], dummy_in.cpu()[0][e_val - 1])
                    trial += 1
            else:
                # generated_sequence = dummy_in
                dummy_in[0][e_val] = next_token
                e_val += 1

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

def adjustmeasure(beat_collect):
    """calculate the total measure length"""
    measure_length = 0
    for b, beat_val in enumerate(beat_collect[1:]):
        measure_length += 4 / getnotetype(beat_val)[0]
    return math.ceil(measure_length)

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

    song_bend_1 = gp.parse('./gprofiles/bend_1.gp5')
    song_bend_2 = gp.parse('./gprofiles/bend_2.gp5')
    song_bend_3 = gp.parse('./gprofiles/bend_3.gp5')
    song_bend_4 = gp.parse('./gprofiles/bend_4.gp5')
    song_bend_5 = gp.parse('./gprofiles/bend_5.gp5')
    song_bend_6 = gp.parse('./gprofiles/bend_6.gp5')
    song_bend_7 = gp.parse('./gprofiles/bend_7.gp5')
    bend1_beat = song_bend_1.tracks[0].measures[0].voices[0].beats[0]
    bend2_beat = song_bend_2.tracks[0].measures[0].voices[0].beats[1]
    bend3_beat = song_bend_3.tracks[0].measures[0].voices[0].beats[0]
    bend4_beat = song_bend_4.tracks[0].measures[0].voices[0].beats[1]
    bend5_beat = song_bend_5.tracks[0].measures[0].voices[0].beats[0]
    bend6_beat = song_bend_6.tracks[0].measures[0].voices[0].beats[0]
    bend7_beat = song_bend_7.tracks[0].measures[0].voices[0].beats[0]

    song_slide_1 = gp.parse('./gprofiles/slide_1.gp5')
    song_slide_2 = gp.parse('./gprofiles/slide_2.gp5')
    song_slide_3 = gp.parse('./gprofiles/slide_3.gp5')
    song_slide_4 = gp.parse('./gprofiles/slide_4.gp5')
    song_slide_5 = gp.parse('./gprofiles/slide_5.gp5')
    song_slide_6 = gp.parse('./gprofiles/slide_6.gp5')
    slide1_beat = song_slide_1.tracks[0].measures[0].voices[0].beats[0]
    slide2_beat = song_slide_2.tracks[0].measures[0].voices[0].beats[0]
    slide3_beat = song_slide_3.tracks[0].measures[0].voices[0].beats[0]
    slide4_beat = song_slide_4.tracks[0].measures[0].voices[0].beats[0]
    slide5_beat = song_slide_5.tracks[0].measures[0].voices[0].beats[0]
    slide6_beat = song_slide_6.tracks[0].measures[0].voices[0].beats[0]

    song_dead = gp.parse('./gprofiles/dead.gp5')
    dead_beat = song_dead.tracks[0].measures[0].voices[0].beats[0]

    song_hammer = gp.parse('./gprofiles/hammer.gp5')
    hammer_beat = song_hammer.tracks[0].measures[0].voices[0].beats[0]

    song_vibrato = gp.parse('./gprofiles/vibrato.gp5')
    vibrato_beat = song_vibrato.tracks[0].measures[0].voices[0].beats[0]

    song_harmonic_1 = gp.parse('./gprofiles/harmonic_1.gp5')
    harmonic_1_beat = song_harmonic_1.tracks[0].measures[0].voices[0].beats[0]
    # Create a new Guitar Pro song
    song = gp.models.Song()

    # Set the song's information
    song.title = titlename
    song.artist = "DjentleViBe"
    song.tempo = 120  # Set the tempo
    song.tracks[0].name = "Guitar"
    song.tracks[0].channel.instrument = 30
    
    song.tracks[0].measures[0].timeSignature.numerator = adjustmeasure(beatval)
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
            if l_val != 0:
                l_val -= 1
        elif note == DEAD_NOTE:
            dead_beat.notes[0].value = note_collect[l_val - 1].value
            dead_beat.notes[0].string = note_collect[l_val - 1].string
            dead_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
            voice.beats[l_val - 1] = dead_beat
            continue
        elif note == HAMMER:
            hammer_beat.notes[0].value = note_collect[l_val - 1].value
            hammer_beat.notes[0].string = note_collect[l_val - 1].string
            hammer_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
            voice.beats[l_val - 1] = hammer_beat
            continue
        elif note == VIBRATO:
            vibrato_beat.notes[0].value = note_collect[l_val - 1].value
            vibrato_beat.notes[0].string = note_collect[l_val - 1].string
            vibrato_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
            voice.beats[l_val - 1] = vibrato_beat
            continue
        elif note == HARMONIC_1:
            harmonic_1_beat.notes[0].value = note_collect[l_val - 1].value
            harmonic_1_beat.notes[0].string = note_collect[l_val - 1].string
            harmonic_1_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
            voice.beats[l_val - 1] = harmonic_1_beat
            continue
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
            beat_collect[l_val - 1].effect.isBend = True
            if note == BEND_NOTE_1:
                bend1_beat.notes[0].value = note_collect[l_val - 1].value
                bend1_beat.notes[0].string = note_collect[l_val - 1].string
                bend1_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = bend1_beat
            elif note == BEND_NOTE_2:
                bend2_beat.notes[0].value = note_collect[l_val - 1].value
                bend2_beat.notes[0].string = note_collect[l_val - 1].string
                bend2_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = bend2_beat
            elif note == BEND_NOTE_3:
                bend3_beat.notes[0].value = note_collect[l_val - 1].value
                bend3_beat.notes[0].string = note_collect[l_val - 1].string
                bend3_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = bend3_beat
            elif note == BEND_NOTE_4:
                bend4_beat.notes[0].value = note_collect[l_val - 1].value
                bend4_beat.notes[0].string = note_collect[l_val - 1].string
                bend4_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = bend4_beat
            elif note == BEND_NOTE_5:
                bend5_beat.notes[0].value = note_collect[l_val - 1].value
                bend5_beat.notes[0].string = note_collect[l_val - 1].string
                bend5_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = bend5_beat
            elif note == BEND_NOTE_6:
                bend6_beat.notes[0].value = note_collect[l_val - 1].value
                bend6_beat.notes[0].string = note_collect[l_val - 1].string
                bend6_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = bend6_beat
            elif note == BEND_NOTE_7:
                bend7_beat.notes[0].value = note_collect[l_val - 1].value
                bend7_beat.notes[0].string = note_collect[l_val - 1].string
                bend7_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = bend7_beat
            continue

        elif SLIDE_NOTE_1 <= note <= SLIDE_NOTE_6:
            if note == SLIDE_NOTE_1:
                slide1_beat.notes[0].value = note_collect[l_val - 1].value
                slide1_beat.notes[0].string = note_collect[l_val - 1].string
                slide1_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = slide1_beat
            elif note == SLIDE_NOTE_2:
                slide2_beat.notes[0].value = note_collect[l_val - 1].value
                slide2_beat.notes[0].string = note_collect[l_val - 1].string
                slide2_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = slide2_beat
            elif note == SLIDE_NOTE_3:
                slide3_beat.notes[0].value = note_collect[l_val - 1].value
                slide3_beat.notes[0].string = note_collect[l_val - 1].string
                slide3_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = slide3_beat
            elif note == SLIDE_NOTE_4:
                slide4_beat.notes[0].value = note_collect[l_val - 1].value
                slide4_beat.notes[0].string = note_collect[l_val - 1].string
                slide4_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = slide4_beat
            elif note == SLIDE_NOTE_5:
                slide5_beat.notes[0].value = note_collect[l_val - 1].value
                slide5_beat.notes[0].string = note_collect[l_val - 1].string
                slide5_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = slide5_beat
            elif note == SLIDE_NOTE_6:
                slide6_beat.notes[0].value = note_collect[l_val - 1].value
                slide6_beat.notes[0].string = note_collect[l_val - 1].string
                slide6_beat.notes[0].beat.duration.value = beat_collect[k_val - 1].duration.value
                voice.beats[l_val - 1] = slide6_beat
            continue
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
            if l_val != MAX_SEQ_LENGTH - 1:
                l_val += 1
    
    return song

def writegpro(filename, song):
    """write gpro file to disk"""
    # Save the song to a Guitar Pro file
    with open("./RESULTS/" + "/" + BACKUP + "/" + filename + ".gp5", 'wb') as file:
        gp.write(song, file)

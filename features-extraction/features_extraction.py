import numpy as np
import sounddevice as sd
import librosa
from scipy import stats
import pitch_extraction as pe

def feature_extraction(y, sr=22050):
    
    # tempo    
    tempo = librosa.beat.tempo(y, sr=sr) # feature
    
    # song length
    length = len(y) / sr # feature

    # signal energy
    energy = np.mean(librosa.feature.rmse(y)) # feature
    
    # harmonic energy ratio 
    harmonic = librosa.effects.harmonic(y)
    energy_ratio = np.mean(librosa.feature.rmse(harmonic)) / energy # feature
    
    # principal pitch
    pitch_list = pe.chroma2pitch(librosa.feature.chroma_stft(harmonic, sr))
    rhythm, melody = pe.rhythm_melody_extract(pitch_list)
    rhythm, melody = pe.convert_rhythm(rhythm, melody, sr, tempo)
    pitch_count = pe.count_pitch(melody)
    main_pitches = pe.get_main_pitch(pitch_count)
    principal_pitch = main_pitches[0] # feature
    
    # principal intervals
    main_intervals = pe.get_main_interv(main_pitches) # feature
    
    # main notes ratio
    main_notes_ratio = np.sum([pitch_count[i] for i in range(12) if i in main_pitches]) / np.sum(pitch_count) # feature
    
    # note changing
    interval_list = pe.pitch_deriv(pitch_list)
    interval_count = pe.count_pitch(interval_list) # feature
    
    # interval changing
    interval_changin_list = pe.pitch_deriv(interval_list)
    interval_changing_count = pe.count_pitch(interval_changin_list) # feature
    
    # rhythm changing
    rhythm_graph = pe.rhythm_graph(rhythm)
    rhythm_graph_flat = [j for i in rhythm_graph for j in i] # feature

    # melody changing
    melody_graph = pe.melody_graph(melody, principal_pitch)
    melody_graph_flat = [j for i in melody_graph for j in i] # feature    
    
    # harmonic auto correlation frequency peak
    #acf_peak = pe.acf_peak(harmonic, sr, tempo) # feature    
    
    # normalization
    interval_changing_count /= np.sum(interval_changing_count)
    interval_count /= np.sum(interval_count)
    melody_graph_flat /= np.sum(melody_graph_flat)
    rhythm_graph_flat /= np.sum(rhythm_graph_flat)
    
    return list(tempo) + [length, energy, energy_ratio, principal_pitch] + main_intervals + [main_notes_ratio] + list(interval_count) + list(interval_changing_count) + list(rhythm_graph_flat) + list(melody_graph_flat) #+ [acf_peak]
#
#def vec_shape(x):
#    return [i for o in x for i in o]
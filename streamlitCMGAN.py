import argparse
import io
import os
import time
import wave

import numpy as np
import pyaudio
import sounddevice as sd
import soundfile as sf
import torchaudio

from pydub import AudioSegment
from natsort import natsorted

from models import generator
from tools.compute_metrics import compute_metrics
from utils_ss_cuda  import *

import librosa
import streamlit as st

#torch.cuda.empty_cache()

@torch.no_grad()
def play_audio(audio_data, sample_rate=16000):
    st.audio(audio_data, format="audio/wav", start_time=0, sample_rate=sample_rate)

def add_noise_to_audio(speech, noise, snr):
    # Calculer le nombre de répétitions nécessaires pour couvrir la longueur du discours
    num_repetitions = int(np.ceil(len(speech) / len(noise)))

    # Répéter le bruit
    repeated_noise = np.tile(noise, num_repetitions)

    # Tronquer le bruit pour correspondre à la longueur du discours
    repeated_noise = repeated_noise[:len(speech)]

    # Calculer la puissance du signal et du bruit
    power_speech = np.sum(speech ** 2) / len(speech)
    power_noise = np.sum(repeated_noise ** 2) / len(repeated_noise)

    # Vérifier si la puissance du bruit est non nulle
    if power_noise > 0:
        # Calculer la puissance du bruit pour atteindre le SNR désiré
        target_power_noise = power_speech / (10 ** (snr / 10))
        scale_factor = np.sqrt(target_power_noise / power_noise)

        if not np.isnan(scale_factor):
            # Effectuer la multiplication
            adjusted_noise_array = repeated_noise * scale_factor
        
            # Additionner le discours et le bruit
            result = speech + adjusted_noise_array

        else:
            st.write('it is Nan')
            # Si scale_factor est NaN, renvoyer le discours seul
            result = speech
    else:
        st.write('power_noise is zero or negative')
        # Si la puissance du bruit est nulle ou négative, renvoyer le discours seul
        result = speech

    return result

# Streamlit app
st.set_page_config(layout="wide")
st.title("Test app")

option = st.sidebar.radio("**Choose an option :**", ("Use existing file", "Record sound"))

if option == "Record sound":
    # duration = st.slider("Select recording duration (seconds)", min_value=1, max_value=10, value=3)
    duration=4 
    sample_rate = 16000
    st.info(f"Click the button to start recording. Recording duration: {duration} seconds.")
    if st.sidebar.button('**Start recording and use CMGAN**'):

    
        # record_button = st.button("Start Recording")
    
        recorded_audio = None
    
    # if record_button:
        # Record audio using sounddevice
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
        sd.wait()
        
        # st.success("Recording completed!")
        
        recorded_audio = audio_data.flatten()
        
        # if recorded_audio is not None:
        # Save recorded audio to a temporary file
        # temp_file_path = save_audio_to_temp_file(recorded_audio)
        rec_file_path = "rec.wav"
        sf.write(rec_file_path, recorded_audio, samplerate=16000)
        # Play the recorded audio
        st.markdown("**Play Recorded Audio**")
        play_audio(recorded_audio)
        
        start_time = time.time()
        model_path = './best_ckpt/ckpt'
        save_tracks = True
        saved_dir = './audio_enregistré'
        n_fft = 400
        
        model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        rec_audio, length = enhance_one_track(model, rec_file_path, saved_dir, 16000 * 16, n_fft, n_fft // 4, save_tracks)
        # Afficher le résultat de la suppression de bruit
        st.markdown("**Audio cleaned with CMGAN**")
        st.audio(rec_audio, format="audio/wav", start_time=0, sample_rate=16000)
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Execution time : {elapsed_time:.2f} seconds")
            
else:
    # speech_file = st.file_uploader("**Choisissez le fichier audio du discours (format WAV)**", type=["wav"])
    # noise_file = st.file_uploader("**Choisissez le fichier audio du bruit (format WAV)**", type=["wav"])
    
    option_speech = ['speech']
    option_noise = ['applause', 'bar', 'course','discussion', 'forest', 'parc','party', 'storm']
    speech_file = st.sidebar.selectbox('**Choose clean speech**', option_speech)
    noise_file = st.sidebar.selectbox('**Choose an noise**', option_noise)
    
    speech_path='./datawav/'+speech_file+'.wav'
    noise_path='./datawav/'+noise_file+'.wav'
    
    snr = st.sidebar.slider("**Choose the signal noise ration (SNR)**", -20, 20, 7)
    
    if speech_file is not None and noise_file is not None:
           # Charger les données audio en tant que tableaux numpy
           # speech, sr_speech = librosa.load(io.BytesIO(speech_file.read()), sr=16000)
           # noise, sr_noise = librosa.load(io.BytesIO(noise_file.read()), sr=16000)
           
           speech, sr_speech = librosa.load(speech_path, sr=16000)
           noise, sr_noise = librosa.load(noise_path, sr=16000)
           
           st.markdown("**Play clean speech**")
           play_audio(speech)
           
           st.markdown("**Play noise**")
           play_audio(noise)
           # Ajouter du bruit au discours
           result = add_noise_to_audio(speech, noise, snr)
           
           # Enregistrer le résultat dans un fichier temporaire
           result_file_path = "result.wav"
           sf.write(result_file_path, result, samplerate=16000)
           
           # Afficher le résultat
           st.markdown("**Noised audio**")
           st.audio(result_file_path, format="audio/wav", start_time=0)
           
           if st.button('Utiliser CMGAN'):
               start_time = time.time()
               model_path = './best_ckpt/ckpt'
               save_tracks = True
               saved_dir = './audio_généré'
               n_fft = 400
               
               model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1)
               model.load_state_dict(torch.load(model_path, map_location='cpu'))
               model.eval()
               est_audio, length = enhance_one_track(model, result_file_path, saved_dir, 16000 * 16, n_fft, n_fft // 4, save_tracks)
               # Afficher le résultat de la suppression de bruit
               st.markdown("**Audio cleaned with CMGAN**")
               st.audio(est_audio, format="audio/wav", start_time=0, sample_rate=16000)
               end_time = time.time()
               elapsed_time = end_time - start_time
               st.write(f"Temps d'exécution : {elapsed_time:.2f} secondes")

"""
    File: modify_audio_clips.py

    Description: Given an input dataset, it downsample audio clips to 16kHz and converts audio to mono when required

    Assumption: Audio clips downloaded

    Input: Directory path where input dataset is located, directory path where all audio clips are stored, modified audio clips directory path where audio clips updated are to be stored
           
    Output: Downsample audio clips and convert audio to mono

    Run: 
        python3 modify_audio_clips.py --input_dataset=/path/to/input/dataset --audio_clip_dir=/path/to/audio_clip_directory --modified_audio_clip_dir=/path/to/modified_audio_clip_directory

    examples: 
        python3 modify_audio_clips.py --input_dataset=./dataset/dataset1.csv --audio_clip_dir=../../AudioClips_DataRepoService/ --modified_audio_clip_dir=./modified_audio_clips/
        python3 modify_audio_clips.py --input_dataset=./dataset/dataset2.csv --audio_clip_dir=../../AudioClips_DataRepoService/ --modified_audio_clip_dir=./modified_audio_clips/
        python3 modify_audio_clips.py --input_dataset=./dataset/dataset3.csv --audio_clip_dir=../../AudioClips_DataRepoService/ --modified_audio_clip_dir=./modified_audio_clips/
"""
   
import pandas as pd
import os
import audio_metadata
import soundfile as sf
from scipy import signal
import librosa
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_dataset', type=str, help="Input dataset file path")
parser.add_argument('-a', '--audio_clip_dir', type=str, help="Path where all audio clips are stored")
parser.add_argument('-m', '--modified_audio_clip_dir', type=str, help="Path where all updated audio clips will be stored")

args = parser.parse_args()

# read input dataset to fetch audio clips to be updated
input_file = args.input_dataset
input_df = pd.read_csv(input_file)
af = list(input_df['audio_clip_file_name'])
print(len(af))

# list path where all audio clips are stored
audio_clip_dir = args.audio_clip_dir

# list path where all modified audio clips are to be stored
modified_audio_clip_dir = args.modified_audio_clip_dir
if not os.path.exists(modified_audio_clip_dir):
    os.makedirs(modified_audio_clip_dir)
modified_audio_clips = os.listdir(modified_audio_clip_dir)

count = 0

for f in af:
    
    count +=1

    audio_file_name = f
    audio_file = audio_clip_dir + audio_file_name

    # extracting audio metadata - frame rate, number of channels, duration
    wave_file = audio_metadata.load(audio_file)
    frame_rate = wave_file['streaminfo'].sample_rate
    num_channels = wave_file['streaminfo'].channels
    duration = wave_file['streaminfo'].duration
    print(f"--File{count}: Audio frame_rate={frame_rate}, channels={num_channels}, duration={duration}")

    audio_path = f"{modified_audio_clip_dir}{audio_file_name}"

    if f not in modified_audio_clips:
        
        # Step1: convert audio to mono 
        if num_channels != 1:

            print("Converting audio to mono:", audio_file_name)
            # Load stereo audio
            stereo_audio, sample_rate = librosa.load(audio_file, sr=None, mono=False)
            # Convert to mono
            mono_audio = librosa.to_mono(stereo_audio)
            # Save mono audio
            sf.write(audio_path, mono_audio, sample_rate)

        # Step2: downsample audio
        if frame_rate != 16000:

            if audio_file_name in modified_audio_clips:
                audio_file = modified_audio_clip_dir + audio_file_name

            print("Downsampling audio:", audio_file_name)
            y, sr = librosa.load(audio_file, sr=None)  # Automatically detects the original sample rate

            # Resample to 16 kHz
            target_sample_rate = 16000   # WhisperFeatureExtractor was trained using a sampling rate of 16000
            y_resampled = signal.resample(y, int(len(y) * target_sample_rate / sr))

            # Save the downsampled audio to a new file
            # audio_path = f"{updated_audio_file_path}{audio_file_name}"
            sf.write(audio_path, y_resampled, target_sample_rate)

"""
    File: google_transcribe.py

    Description: Transcribe audio files recorded by describers on YouDescribe platform using google cloud Speech to Text api

    Assumptions: Downloaded all audio clips to transcribe

    Input: Audio Clip directory path to transcribe

    Output: Audio Clip Transcription Results

    Run: python3 google_transcribe.py --directory_path=/path/to/directory

     example:
        python3 google_transcribe.py --directory_path /Users/Koob/Documents/YouDescribe/AudioClips_DataRepoService/
"""

import os
import audio_metadata
import json
from google.protobuf.json_format import MessageToDict
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file
load_dotenv()

google_application_credentials = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory_path', type=str, help="Audio clips directory path")

args = parser.parse_args()
directory = args.directory_path

def frame_rate_channel(audio_file_name):
    print("--Extracting Audio metadata--")
    wave_file = audio_metadata.load(audio_file_name)
    frame_rate = wave_file['streaminfo'].sample_rate
    channels = wave_file['streaminfo'].channels
    print("--Audio frame_rate={} and channels={}--".format(frame_rate,channels))
    return frame_rate,channels
    
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

def google_transcribe():

    dir_list = os.listdir(directory)

    transcriptsList = []

    for audio_file_index in range(2):
        print("File number:", audio_file_index)
        audio_file_name = dir_list[audio_file_index]
        print(audio_file_name)
        filepath = directory
        file_name = filepath + audio_file_name

        print("===========")
        print(file_name)
        print("============")

        try:
            frame_rate, channels = frame_rate_channel(file_name)
            print("frame rate:", frame_rate)
            print("channels:", channels)
        except Exception as e:
            print(f"An exception occured while fetching audio metadata: {e}")
            f_audioMetadata = open('./audiometadataException.txt',"a")
            f_audioMetadata.write(audio_file_name + "\n")
            f_audioMetadata.close()
            continue

        bucket_name = 'speechtotext-transcripts'
        source_file_name = filepath + audio_file_name
        destination_blob_name = audio_file_name

        print("===========\nUploading to Google Bucket\n============")
 
        upload_blob(bucket_name, source_file_name, destination_blob_name)
        
        gcs_uri = 'gs://'+bucket_name+'/' + audio_file_name
        transcript = ''
        
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(uri=gcs_uri)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            audio_channel_count= channels,
            sample_rate_hertz=frame_rate,
            use_enhanced=True,
            enable_speaker_diarization=True,
            language_code='en-US',
            model='video',
        )

        # Detects speech in the audio file
        print("===========\nUploading to Speech to text\n============")

        try:
            operation = client.long_running_recognize(config=config, audio=audio)

            response = operation.result(timeout=10000)

            if response.results:

                print("===========\nResponse from Speech to Text\n===========")
    
                transcript = "".join(result.alternatives[0].transcript for result in response.results)

                transcriptsDict = {}
                transcriptsDict['audio_filename'] = audio_file_name
                transcriptsDict['transcription_response'] = MessageToDict(response._pb)
                transcriptsDict['transcript'] = transcript
                transcriptsList.append(transcriptsDict)

                # Storing filename and transcripts            
                f_transcript = open('./transcriptionResults.txt',"a")
                f_transcript.write(audio_file_name + ' ' + transcript + "\n")
                f_transcript.close()

            else:
                # untranscribed audio files
                f_untranscribed = open('./untranscribedFiles.txt',"a")
                f_untranscribed.write(audio_file_name+"\n")
                f_untranscribed.close()

        except Exception as e:
            print(f"An exception occured: {e}")
            f_exception = open('./untranscribedExceptionFiles.txt',"a")
            f_exception.write(audio_file_name+"\n")
            f_exception.close()

        
        print("============\nDeleting Data from Bucket\n============")

        delete_blob(bucket_name, destination_blob_name)

        print("===========\nDeleted Data from Bucket===========")

    with open('./transcriptionResponse.json', 'w') as f_response_json:
        json.dump(transcriptsList, f_response_json, indent=4)

if __name__ == '__main__':
    google_transcribe()
    
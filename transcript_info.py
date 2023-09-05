"""
    File: transcript_info.py

    Description: Get start_time, end_time, duration for each audio clip transcribed using google cloud Speech to Text api

    Assumptions: Transcriptions generated with google_transcribe.py

    Output: Dataframe with above mentioned columns

    Run: python3 transcript_info.py
"""

import pandas as pd
import numpy as np

def transcriptInfo():

    tr = pd.read_json('./transcriptionResponse.json')

    tr['transcript'].replace('', np.nan, inplace=True)
    tr.dropna(subset=['transcript'], inplace=True)
    tr.reset_index(drop=True, inplace=True)

    tr['transcript_start_time'] = tr['transcription_response'].apply(lambda x: pd.json_normalize(x['results'][-1]['alternatives'][0])['words'].iloc[0][0]['startTime']) 
    tr['transcript_end_time'] = tr['transcription_response'].apply(lambda x: pd.json_normalize(x['results'][-1]['alternatives'][0])['words'].iloc[0][-1]['endTime'])

    tr['transcript_start_time'] = tr['transcript_start_time'].str.replace('s',"")
    tr['transcript_end_time'] = tr['transcript_end_time'].str.replace('s',"")

    tr['transcript_start_time'] = tr['transcript_start_time'].astype(float)
    tr['transcript_end_time'] = tr['transcript_end_time'].astype(float)
    tr['transcript_duration'] = tr['transcript_end_time'] - tr['transcript_start_time']

    tr.drop(['transcription_response'], axis=1, inplace=True)

    tr.to_csv('./ggl_stt_transcriptions.csv', index=False, header=True, encoding='utf-8')

if __name__ == '__main__':
    transcriptInfo()
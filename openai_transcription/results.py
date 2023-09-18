"""
    File: results.py

    Description: Given an input dataset, it will result in an additional column with just transcript sentence

    Assumption: Run openai_transcribe_wlv2.py first

    Input: Directory path where input dataset is located, directory path to output dataset
           
    Output: Audio Clip transcription results

    Run: 
      python3 results.py --input_dataset=/path/to/input/dataset --output_dataset=/path/to/output/dataset

    examples: 
      python3 results.py --input_dataset=./results/dataset_whisper_large_v2_exception.csv --output_dataset=./results/dataset_whisper_large_v2_exception_updated.csv
      python3 results.py --input_dataset=./results/dataset1_whisper_large_v2.csv --output_dataset=./results/dataset1_whisper_large_v2_updated.csv
      python3 results.py --input_dataset=./results/dataset2_whisper_large_v2.csv --output_dataset=./results/dataset2_whisper_large_v2_updated.csv
      python3 results.py --input_dataset=./results/dataset3_whisper_large_v2.csv --output_dataset=./results/dataset3_whisper_large_v2_updated.csv
"""

import pandas as pd
import ast
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input_dataset', type=str, help="Input dataset file path")
parser.add_argument('-o', '--output_dataset', type=str, help="Output dataset file path")

args = parser.parse_args()

input_dataset = args.input_dataset
output_dataset = args.output_dataset

df = pd.read_csv(input_dataset)

# convert string to list for column 'oa_transcript'
df['oa_transcript'] = df['oa_transcript'].apply(lambda x: ast.literal_eval(x))

# concatenate sentences from transcript data
def concatenate_sentences(dicts_list):
    return ''.join([item['sentence'] for item in dicts_list])

df['oa_transcript_sentence'] = df['oa_transcript'].apply(concatenate_sentences)

# redorder columns 
column_order = list(df.columns)
column_order.remove('oa_transcript')
column_order.insert(column_order.index('oa_transcript_sentence')+1, 'oa_transcript')

df = df[column_order]

df.to_csv(output_dataset, index=False, header=True)
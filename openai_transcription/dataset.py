"""
    File: dataset.py

    Description: Given an input dataset, it will divide dataset into 3 parts

    Input: Directory path where input dataset is located
           
    Output: Divided dataset into 3 parts

    Run: 
      python3 dataset.py --input_dataset=/path/to/input/dataset
      
    example:
      python3 dataset.py --input_dataset=../../YD_2.0_transcribed_audio_clips_english_v5.csv

"""

import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dataset', type=str, help="Input dataset file path")
args = parser.parse_args()

# read input dataset to fetch audio clips to be updated
input_file = args.input_dataset

df = pd.read_csv(input_file)

# dividing dataset into 3 parts
size = len(df)//3
df1 = df[:size]
df2 = df[size:2*size]
df3 = df[2*size:]

df1.to_csv('./dataset/dataset1.csv', index=False, header=True)
df2.to_csv('./dataset/dataset2.csv', index=False, header=True)
df3.to_csv('./dataset/dataset3.csv', index=False, header=True)
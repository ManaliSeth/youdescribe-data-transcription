"""
    File: evaluate_performance.py

    Description: Evaluating performance of transcriptions by Listen By Code / Google Cloud Speech to Text API and Open AI whisper-large-v2 model

    Input: Directory where input dataset is located
           
    Output: Directory where output dataset should be located with evaluation metric results

    Run: python3 evaluate_performance.py --input_dataset=/path/to/input/dataset --output_dataset=/path/to/output/dataset
      example:
        within open_ai transcription foler, run below command in terminal
        python3 evaluate_performance.py --input_dataset1=./results/dataset1_whisper_large_v2_updated.csv --input_dataset2=./results/dataset2_whisper_large_v2_updated.csv --input_dataset3=./results/dataset3_whisper_large_v2_updated.csv --input_dataset4=./results/dataset_whisper_large_v2_exception_updated.csv  --output_dataset=./results/dataset_whisper_large_v2_evaluation.csv 
"""

import pandas as pd
import argparse
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument('-i1', '--input_dataset1', type=str, help="Input dataset file path")
parser.add_argument('-i2', '--input_dataset2', type=str, help="Input dataset file path")
parser.add_argument('-i3', '--input_dataset3', type=str, help="Input dataset file path")
parser.add_argument('-i4', '--input_dataset4', type=str, help="Input dataset file path")
parser.add_argument('-o1', '--output_dataset', type=str, help="Output evaluation file path")

args = parser.parse_args()
input_file1 = args.input_dataset1
input_file2 = args.input_dataset2
input_file3 = args.input_dataset3
input_file4 = args.input_dataset4
output_file = args.output_dataset

# evaluation metrics - bleu score
def generate_bleu_score(sentence1, sentence2):

    # Create a SmoothingFunction
    smooth_fn = SmoothingFunction().method1

    bleu_score = sentence_bleu(sentence1, sentence2, smoothing_function=smooth_fn)
    bleu_score = round(bleu_score, 3)

    return bleu_score

# evaluation metrics - cosine similarity
def calculate_cosine_similarity(sentence1, sentence2):

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the vectorizer on the texts
    vectorized_texts = vectorizer.fit_transform([sentence1, sentence2])

    # Calculate the cosine similarity between the vectorized texts
    similarity = cosine_similarity(vectorized_texts[0], vectorized_texts[1])[0][0]
    similarity = round(similarity, 3)

    return similarity

def evaluate_transcripts():

    # Load dataset with audio clip transcripts
    input1_df = pd.read_csv(input_file1)
    input2_df = pd.read_csv(input_file2)
    input3_df = pd.read_csv(input_file3)
    input4_df = pd.read_csv(input_file4)

    input_df = pd.concat([input1_df, input2_df, input3_df, input4_df])
    input_df = input_df.reset_index(drop=True)

    # output file path to store evaluation results
    output_file_dir = './results'
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)

    output_file_path = output_file
    if not os.path.isfile(output_file_path):
        data = {'youdescribe_link': [], 'audio_clip_file_path':[], 'audio_clip_file_name':[], 'audio_clip_duration':[], 'audio_clip_start_time':[], 'audio_clip_end_time':[], 'audio_clip_playback_type':[], 'db_transcript':[], 'oa_transcript_sentence':[], 'bleu_score':[], 'cosine_similarity':[]}
        output_df = pd.DataFrame(data)
        output_df.to_csv(output_file_path, index=False, header=True)
    else:
        output_df = pd.read_csv(output_file_path)
    
    count, save_interval = 0,5

    for row in range(len(input_df)):

        try:
            print(row)
            count+=1
            sentence1 = input_df.loc[row]['db_transcript']
            sentence2 = input_df.loc[row]['oa_transcript_sentence']
            
            bleu_score = generate_bleu_score([sentence1], sentence2)
            cosine_similarity = calculate_cosine_similarity(sentence1, sentence2)

            # Fetching info about the audio clip
            youdescribe_link = input_df.loc[row]['youdescribe_link']
            file_path = input_df.loc[row]['audio_clip_file_path']
            file_name = input_df.loc[row]['audio_clip_file_name']
            duration = input_df.loc[row]['audio_clip_duration']
            start_time = input_df.loc[row]['audio_clip_start_time']
            end_time = input_df.loc[row]['audio_clip_end_time']
            playback_type = input_df.loc[row]['audio_clip_playback_type']
            db_transcript = input_df.loc[row]['db_transcript']
            oa_transcript_sentence = input_df.loc[row]['oa_transcript_sentence']

            new_data = [{
            'youdescribe_link': youdescribe_link,
            'audio_clip_file_path': file_path,
            'audio_clip_file_name': file_name,
            'audio_clip_duration': duration,
            'audio_clip_start_time': start_time,
            'audio_clip_end_time': end_time,
            'audio_clip_playback_type': playback_type,
            'db_transcript': db_transcript,
            'oa_transcript_sentence': oa_transcript_sentence,
            'bleu_score': bleu_score,
            'cosine_similarity': cosine_similarity
            }]

            new_row_df = pd.DataFrame(new_data)

            # Append the new row
            output_df = pd.concat([output_df, new_row_df], ignore_index=True)
            
            if count % save_interval == 0:
                output_df.to_csv(output_file_path, index=False)
        
        except Exception as e:
            with open('./results/exception_evaluation.txt', 'a') as exp:
                exp.write(file_name + ' - ' + str(e) + '\n')
                exp.close()
        
    output_df.to_csv(output_file_path, index=False)

# Evatuate transcriptions
evaluate_transcripts()
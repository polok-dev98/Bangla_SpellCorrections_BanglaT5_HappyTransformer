#import library
import pandas as pd
import numpy as np
import random
import csv
#---------------------------------training dataset building------------------------------------------------
# Load data from input CSV file
t_df = pd.read_csv("main_dataset_v3.csv")

# Select 20000 random rows from the input dataframe
selected_rows_tr = t_df.iloc[0:20000,:]

# Extract Bangla sentences from the selected rows
bangla_sentences_tr = selected_rows_tr['Sentence'].tolist()

# List of Bengali consonant characters for misspelling
consonant_characters = [
    'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট',
    'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ',
    'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়','অ','আ','ই','উ']

# Function to randomly misspell some words
def misspell(sentence, probability=0.3):
    words = sentence.split()
    for i in range(len(words)):
        if random.random() < probability:
            # Misspell the word by changing a random character
            word = list(words[i])
            random_index = random.randint(0, len(word) - 1)
            random_consonant = random.choice(consonant_characters)
            word[random_index] = random_consonant
            words[i] = ''.join(word)
    return ' '.join(words)

# Generate misspelled sentences
misspelled_sentences_tr = [misspell(sentence) for sentence in bangla_sentences_tr]

# Create DataFrame with misspelled sentences
df_tr = pd.DataFrame({'sentence': misspelled_sentences_tr})

# Add corrections column with original sentences
df_tr['corrections'] = bangla_sentences_tr

# Save DataFrame to CSV
df_tr.to_csv('train_dataset.csv', index=False)

#---------------------------------validation dataset building------------------------------------------------

# Load data from input CSV file
e_df = pd.read_csv("main_dataset_v3.csv")

# Select 1500 random rows from the input dataframe
selected_rows_ev = e_df.iloc[20001:250001,:]

# Extract Bangla sentences from the selected rows
bangla_sentences_ev = selected_rows_ev['Sentence'].tolist()

# List of Bengali consonant characters for misspelling
consonant_characters = [
    'ক', 'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট',
    'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ',
    'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়','অ','আ','ই','উ']

# Function to randomly misspell some words
def misspell(sentence, probability=0.3):
    words = sentence.split()
    for i in range(len(words)):
        if random.random() < probability:
            # Misspell the word by changing a random character
            word = list(words[i])
            random_index = random.randint(0, len(word) - 1)
            random_consonant = random.choice(consonant_characters)
            word[random_index] = random_consonant
            words[i] = ''.join(word)
    return ' '.join(words)

# Generate misspelled sentences
misspelled_sentences_ev = [misspell(sentence) for sentence in bangla_sentences_ev]

# Create DataFrame with misspelled sentences
df_ev = pd.DataFrame({'sentence': misspelled_sentences_ev})

# Add corrections column with original sentences
df_ev['corrections'] = bangla_sentences_ev

# Save DataFrame to CSV
df_ev.to_csv('eval_dataset.csv', index=False)

# load the train
train_dataset = pd.read_csv("train_dataset.csv")
# load the eval data
eval_dataset = pd.read_csv("eval_dataset.csv")

#-------------Convert our train and validation dataset into happyTransformer model's train and validation dataset format-------------------------


def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:  # Specify encoding as 'utf-8'
        writer = csv.writer(csvfile)
        writer.writerow(["input", "target"])
        for index, row in dataset.iterrows():
            # Assuming each row contains "sentence" and "corrections" columns
            input_text = "grammar: " + row["sentence"]
            correction = row["corrections"]
            if input_text and correction:
                writer.writerow([input_text, correction])

generate_csv("train.csv", train_dataset)
generate_csv("eval.csv", eval_dataset)


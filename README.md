# Bangla Spell Correction using BanglaT5
</br>

<div align="center">
    ![logo](https://github.com/polok-dev98/Bangla_SpellCorrections_BanglaT5_HappyTransformer/assets/104290708/626df27b-16df-4e1a-af54-f20605066f4d)
</div>


</br>
</br>

This repository contains code for a project focused on Bangla spell correction using the BanglaT5 model. The project involves preprocessing data, training the model, and testing its performance. Here the main goal is to fine tune bangla T5 model for Bangla spell correction task.

## Dataset
Here used a Bangla text corpus available on Kaggle ([link](https://www.kaggle.com/datasets/nuhashafnan/corpus)) to build the dataset for training the model. Each sentence in the dataset was deliberately misspelled to create a training set for the model.


## Usage
1. **Preprocess the Dataset**: Run `preprocess.py` to prepare the dataset for training. This script randomly selects a subset of the dataset, introduces misspellings, and saves it to a new CSV file.

2. **Train the Model**: Execute `train.py` to train the BanglaT5 model using the preprocessed dataset. The script utilizes the `HappyTextToText` module for training. Adjust training parameters such as batch size and number of epochs as needed.

3. **Test the Trained Model**: Run `test.py` to evaluate the trained model on a separate test dataset. This script loads the trained model, performs inference on new text samples, and prints the corrected text.

4. **Refer to the Notebook**: For a detailed walkthrough of the entire process, refer to the `notebook.ipynb` file.


## Files
1. **Preprocess the Dataset**: 
    - The script `preprocess.py` imports necessary libraries such as pandas, numpy, random, and csv.
    - Training Dataset Building:
        - The main dataset (`main_dataset_v3.csv`) is loaded into a pandas DataFrame (`t_df`).
        - 20,000 random rows are selected from the input dataframe to create the training dataset (`selected_rows_tr`).
        - Bangla sentences are extracted from the selected rows and converted to a list (`bangla_sentences_tr`).
        - A list of Bengali consonant characters for misspelling is defined (`consonant_characters`).
        - The `misspell()` function is defined to randomly misspell some words in the sentences.
        - Misspelled sentences are generated based on the defined probability and stored in `misspelled_sentences_tr`.
        - A new DataFrame (`df_tr`) is created with misspelled sentences and their corrections.
        - The DataFrame is saved to a CSV file named `train_dataset.csv`.
    - Validation Dataset Building:
        - Another subset of the main dataset is loaded into a new DataFrame (`e_df`).
        - 15,000 random rows are selected to create the validation dataset (`selected_rows_ev`).
        - Bangla sentences are extracted from the selected rows and converted to a list (`bangla_sentences_ev`).
        - Similar to the training dataset, misspelled sentences are generated and stored in `misspelled_sentences_ev`.
        - A new DataFrame (`df_ev`) is created with misspelled sentences and their corrections.
        - The DataFrame is saved to a CSV file named `eval_dataset.csv`.
    - Converting Datasets:
        - The preprocessed training and validation datasets are loaded (`train_dataset.csv` and `eval_dataset.csv`, respectively).
        - A function named `generate_csv()` is defined to convert the datasets into the format required by the HappyTransformer model.
        - The function writes the data to CSV files (`train.csv` and `eval.csv`) with "input" and "target" columns, where "input" contains the misspelled sentences and "target" contains the corrections.

2. **Train the Model**: 
    - The script `train.py` imports necessary modules from the Happy Transformer library (`TTTrainArgs` and `HappyTextToText`) along with the built-in `pickle` module for object serialization.
    - Loading the Model:
        - The BanglaT5 model is initialized using `HappyTextToText`.
        - The model is instantiated with the model type "T5" and the specific model name "csebuetnlp/banglat5". This step loads the pre-trained BanglaT5 model for further training.
    - Evaluation Before Training:
        - The `eval()` method of `HappyTextToText` is used to evaluate the model's performance before training.
        - The method takes the evaluation dataset file (`eval.csv`) as input and returns evaluation metrics, including the loss.
        - The loss before training is printed to the console.
    - Training the Model:
        - Training arguments (`args`) are defined using `TTTrainArgs`.
        - In this case, a batch size of 1 and 6 epochs are specified for training.
        - The `train()` method of `HappyTextToText` is called to train the model.
        - The training dataset file (`train.csv`) and the defined training arguments (`args`) are provided as inputs.
    - Evaluation After Training:
        - After training, the model is evaluated again on the same evaluation dataset (`eval.csv`) using the `eval()` method.
        - The loss after training is printed to the console.
    - Saving the Trained Model:
        - The trained model (`happy_tt`) is serialized and saved to a file named `model.pkl` using the `pickle.dump()` function.
        - This allows the trained model to be reused later without retraining.

3. **Test the Trained Model**: 
    - The script `test.py` loads the trained model from the serialized file `model.pkl` using the `pickle.load()` function.
    - Setting Beam Search Parameters:
        - Beam search is a method used in sequence generation tasks to explore multiple possible sequences.
        - `TTSettings` from the Happy Transformer library is used to define beam search parameters.
        - In this case, `num_beams` is set to 5, indicating that the model will explore 5 different sequences during generation.
        - `min_length` and `max_length` are set to 1 and 32, respectively, to control the minimum and maximum length of the generated sequences.
    - Generating Text:
        - An example sentence in Bangla is provided (`example_1`), formatted with the prefix "grammar:" as expected by the model.
        - The `generate_text()` method of the loaded model (`model`) is called to generate corrected text based on the provided example.
        - The method takes the example sentence and additional arguments such as beam search settings (`args`) to guide the generation process.
        - The generated text result is stored in `result_1`.
    - Printing the Result:
        - The corrected text is extracted from the `result_1` object using the `.text` attribute.
        - Finally, the corrected text is printed to the console using the `print()` function.

4. `requirements.txt`: Lists all the dependencies required to run the code in the repository.

</br>
Reference: [Fine-Tune Grammar Correction](https://www.vennify.ai/fine-tune-grammar-correction/)

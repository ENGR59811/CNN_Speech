# The City College of New York, City University of New York  
# Written by Olga Chsherbakova                                                                   
# October, 2023
# 
# Speech Recognition and Translator System using CNNs

import os
import sounddevice as sd
from CNN_Predict import CNN_Predict
from PLAYBack_Translation import PlayBack_Translation
from CNN_Train import CNN_Train,get_audio_files
from keras import models

# This function is designed to rename a specific type of WAV file, assuming a 
# particular naming convention where "ref.wav" is removed from the original 
# filename. It is useful when you have multiple WAV files following this  
# naming pattern and want to create new filenames with "ref.wav" removed.
def rename_wav_file(input_path):
    try:
        path, filename = os.path.split(input_path)

        _, file_extension = os.path.splitext(filename)

        # Create a new file name ("you_ref.wav")
        new_filename = filename.split("_")
        new_filename = "_".join(new_filename[:-1]) + "_ref.wav"
        # Create new path
        output_path = os.path.join(path, new_filename)
        return "\\".join(output_path.split("\\")[2:])
    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
    except Exception as e:
        print(f"An error occurred while renaming the file: {str(e)}")

def speech_recognition_main():
    # select training directory
    train_dir = 'TrainingSet'
    # select reference directory
    ref_dir = 'ReferenceSet'

    print('WELCOME TO UNIVERSAL TRANSLATOR. MAKE YOUR SELECTION:')
    print('Enter 1 to train a new CNN')
    print('Enter 2 to load an existing CNN')
    user_entry = int(input('Enter your choice: '))

    if user_entry == 1:
        # Train a new CNN
        print('Be patient until statistics are displayed...')
        cnn_trained, audio_files = CNN_Train(train_dir)
        input('CNN training is complete. Press enter to start using CNN.\n')
    elif user_entry == 2:
        # Load an existing CNN
        cnn_file = input('Enter CNN name (with .h5 extension): ')
        try:
            # load the CNN and assign it to cnn_trained
            cnn_trained = models.load_model(cnn_file)
            audio_files = get_audio_files(train_dir)
        except:
            print('Enter a valid file name next time... Leaving... Bye')
            return
    else:
        print('Enter a valid choice next time... Leaving... Bye')
        return
    
    # Now there is a trained CNN loaded (either a new one or an existing one).
    # Use trained CNN to make predictions:
    input('PRESS ENTER AND IMMEDIATELY SAY A WORD INTO YOUR MICROPHONE')
    print('Recording...')
    # Record audio
    # setup recording parameters for microphone
    sample_rate = 16000
    duration = 2  # seconds
    
    # create the object that performs recording:
    rec_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, \
                      channels=1)
    sd.wait()

    # playback the voice recorded:
    print('Replaying recording...')
    sd.play(rec_data, samplerate=sample_rate)
    sd.wait()

    # read languages from reference directory and store them into a list
    languages = [d for d in os.listdir(ref_dir) \
                 if os.path.isdir(os.path.join(ref_dir, d))]
    for idx, lang in enumerate(languages, 1):
        print(f"Enter {idx} for {lang}")

    # select translation output language
    try:
        user_input = int(input('What is your language choice: '))
        output_lang = languages[user_input - 1]
        print(f'You selected {output_lang} as the output language.')
    except:
        print('Your selection is out of bounds... Leaving... Bye')
        return

    # CNN predicts the translation
    cnn_prediction = CNN_Predict(rec_data[:, 0], sample_rate, \
                                 cnn_trained, audio_files)
    # Get the reference file path
    full_ref_path = os.path.join(ref_dir, output_lang, \
                str(cnn_prediction).replace("TrainingSet\\English\\", ""))
    full_ref_path = \
        f"ReferenceSet\\{output_lang}\\" + rename_wav_file(full_ref_path)
    
    # ads: audio data store is an object that holds audio files
    try:
        PlayBack_Translation(rec_data[:, 0], sample_rate, full_ref_path)
    except:
        print(f'translation for {output_lang} unavailable in {cnn_prediction}')

    user_entry = input('Do you want to save CNN? (y/n) ')
    if user_entry == 'y':
        file_name = input('Enter file name (with .h5 extension): ')
        cnn_trained.save(file_name)
    print('Have a predictably nice day!')

speech_recognition_main()
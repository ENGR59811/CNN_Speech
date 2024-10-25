import os
import sounddevice as sd
import numpy as np
import wavio
from pathlib import Path

# select training directory
train_dir = './TrainingSet/'
# select reference directory
ref_dir = './ReferenceSet/'

print('****************************************************')
print('** WELCOME TO THE SPEECH RECOGNITION DATA CREATOR **')
print('****************************************************')

input('\nPress ENTER to start\n\n')

# setup recording parameters for microphone
sample_rate = 16000
bits_per_sample = 8
noof_channels = 1
recording_duration = 2  # seconds

finished = False
while not finished:
    input('PRESS ENTER AND IMMEDIATELY SAY A WORD INTO YOUR MICROPHONE')
    print('Recording...')

    # record the voice
    rec_data = sd.rec(int(recording_duration * sample_rate), samplerate=sample_rate, channels=noof_channels)
    sd.wait()  # wait until recording is finished

    # playback the voice recorded
    print('Replaying recording...')
    sd.play(rec_data, samplerate=sample_rate)
    sd.wait()

    choice = input('Do you want to save this recording? (y/n) \n')
    if choice.lower() == 'y':
        print('\nEnter 1 to save as training data')
        print('Enter 2 to save as a reference\n')
        
        type_choice = input('Enter your choice: ')
        
        if type_choice == '1':
            output_dir = Path(train_dir)
        elif type_choice == '2':
            # read languages from reference directory and store them into a list
            languages = [d.name for d in Path(ref_dir).iterdir() if d.is_dir()]
            
            for idx, lang in enumerate(languages, start=1):
                print(f'Enter {idx} for {lang}')
            print(f'Enter {len(languages) + 1} to add a new language')

            lang_choice = int(input('Select the language of the recording: '))
            if lang_choice == len(languages) + 1:
                language = input('Type the new language to add: ')
                output_dir = Path(ref_dir) / language
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                language = languages[lang_choice - 1]
                output_dir = Path(ref_dir) / language
        else:
            print('Enter a valid option next time... Leaving... Bye')
            break

        # Check or create label directories
        label_dirs = [d.name for d in output_dir.iterdir() if d.is_dir()]
        
        for idx, label in enumerate(label_dirs, start=1):
            print(f'Enter {idx} for {label}')
        print(f'Enter {len(label_dirs) + 1} to add a new label')

        label_choice = int(input('Select a label: '))
        if label_choice == len(label_dirs) + 1:
            label = input('Type the new label to add: ')
            label_dir = output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
        else:
            label = label_dirs[label_choice - 1]
            label_dir = output_dir / label
        
        file_name = input('Enter a file name for the recording: ')
        file_path = label_dir / f'{file_name}.wav'

        # Save recording
        rec_data = np.squeeze(rec_data)  # remove unnecessary dimensions
        wavio.write(str(file_path), rec_data, sample_rate, sampwidth=bits_per_sample // 8)
        
        print('Successfully saved recording as:')
        print(file_path)

    choice = input('Do you want to create another recording? (y/n)\n')
    if choice.lower() != 'y':
        finished = True

print('Have a nice day!')

import numpy as np
from keras import layers, models
import os
from Speech_Spectrograms import speech_spectrograms

# CNN_Train Summary of this function goes here
# Train a new CNN using information from TrainingSet directory
# Then return trained CNN


# load the training audio dataset into ads_train.
# train_dir contains the path to the dataset.
# Function for searching mp3 files in a directory and its subfolders
def get_audio_files(directory):
    mp3_files = []    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                mp3_files.append(os.path.join(root, file))

    return mp3_files


def CNN_Train(train_dir):

    # Load audio files from train_dir
    # audio_files = [f for f in os.listdir(train_dir) if os.path.isfile(f)]
    audio_files = get_audio_files(train_dir)
    # Compute their mel-spectrograms using the speech_spectrograms function
    segment_duration = 2  # Adjust based on your data
    frame_duration = 0.025
    hop_duration = 0.010
    num_bands = 40
    epsil = 1e-6
    # convert the audio samples into images by generating their
    # spectrograms
    X_train = speech_spectrograms(audio_files, segment_duration, \
                                  frame_duration, hop_duration, num_bands)
    X_train = np.log10(X_train + epsil)
    
    # Assign labels based on the order of the files in the directory
    Y_train = np.arange(len(audio_files))

    # append a depth of 1 to specsize to get the input size of the CNN
    input_shape = (num_bands, X_train.shape[2], 1)
    num_classes = len(audio_files)

    # Define the architecture of the CNN
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        
        # create the first convolution layer, it has 12 convolutional
        # filters, each filter has a height and width of 3 and the input is
        # zero padded.
        # the input is padded with the appropriate number of zeros so that
        # the size of the output is the same size as the input.
        layers.Conv2D(12, (3, 3), padding='same'),
        # normalize the output of the convolution between the range -1 
        # and 1. this prevents the updated weights from increasing 
        # explosively during training.
        # layers.BatchNormalization(),
        # apply rectilinear unit activation function (ReLU) that sets all 
        # negative values to zero.
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2), padding='same'),

        layers.Conv2D(24, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2), padding='same'),

        layers.Conv2D(48, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((3, 3), padding='same', strides=(2, 2)),

        layers.Conv2D(48, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Conv2D(48, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((1, 13)),

        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train the CNN
    model.fit(X_train, Y_train, epochs=50, batch_size=32, shuffle=True)

    return model, audio_files
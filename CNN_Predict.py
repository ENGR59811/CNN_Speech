import os
import numpy as np
import soundfile as sf
from Speech_Spectrograms import speech_spectrograms


# CNN_Predict: Use CNN to predict the word 

def CNN_Predict(rec_data, sample_rate, model, audio_files):

    # Save audio data to a temporary file
    temp_file = "temp.wav"
    sf.write(temp_file, rec_data, sample_rate)

    # Compute the mel-spectrogram from the audio data
    # set audio parameters
    # Duration of the entire recording
    segment_duration = len(rec_data) / sample_rate  
    frame_duration = 0.025
    hop_duration = 0.010
    num_bands = 40
    epsil = 1e-6
    
    # compute speech spectrograms, they will be the inputs
    # to the CNN
    X_test = speech_spectrograms([temp_file], segment_duration, \
                                 frame_duration, hop_duration, num_bands)
    X_test = np.log10(X_test + epsil)
    X_test = np.expand_dims(X_test, axis=-1)
    # Use the model to predict the class
    predictions = model.predict(X_test)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Delete the temporary file
    os.remove(temp_file)

    return audio_files[predicted_class]
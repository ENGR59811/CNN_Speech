import librosa
import numpy as np


def speech_spectrograms(files, segment_duration, frame_duration, \
                        hop_duration, num_bands, sr=16000):
    print("Computing speech spectrograms...")

    hop_length = int(hop_duration * sr)

    all_spectrograms = []

    # 1. Calculate the spectrograms for all files
    for file in files:
        y, _ = librosa.load(file, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512,\
                                    hop_length=hop_length, n_mels=num_bands)
        all_spectrograms.append(mel_spec)

    # 2. Determine the maximum width among all spectrograms
    max_width = 419 

    final_spectrograms = []

    # 3. Pad each spectrogram to the maximum width
    for i, mel_spec in enumerate(all_spectrograms):
        spec_width = mel_spec.shape[1]
        left = (max_width - spec_width) // 2
        padded_spec = np.zeros((num_bands, max_width))

        # Diagnostic information
        
        try:
            padded_spec[:, left:left + spec_width] = mel_spec
        except ValueError as e:
            print(f"Error while processing file {files[i]}: {e}")
            continue

        final_spectrograms.append(padded_spec)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} files out of {len(files)}")

    print("...done")

    return np.array(final_spectrograms)

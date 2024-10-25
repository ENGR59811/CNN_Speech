
import sounddevice as sd
import soundfile as sf

def PlayBack_Translation(rec_data, sample_rate, ref_file):

    # Play back the user's recorded word
    
    sd.play(rec_data, samplerate=sample_rate)
    sd.wait()

    # Play the translated word from the reference file
    
    ref_data, ref_sample_rate = sf.read(ref_file)
    sd.play(ref_data, samplerate=ref_sample_rate)
    sd.wait()
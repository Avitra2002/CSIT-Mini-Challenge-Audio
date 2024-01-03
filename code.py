###TASK 1###
# Breaking it down:
# 1) From the hints, I would need to reverse the sounds(maybe I need to chnage it to an array and reverse it)
# 2) basic audio maniputlations like speed up or down might need to be applied
# 3) I believe the output might give me a location
####################
import librosa
import soundfile as sf

# Load the audio file
file_path = "/Users/phonavitra/Desktop/CSIT/CSIT_DS_Mini-Challenge/CSIT_DS_Mini-Challenge/Task_1/T1_audio.wav"
audio, sr = librosa.load(file_path, sr=None)

# Reverse the audio
audio_reversed = audio[::-1]

#Adjust the speed if necessary (you can change the rate to speed up or slow down)
audio_stretched = librosa.effects.time_stretch(audio_reversed, rate=1.0)

# Save the reversed (and possibly speed-adjusted) audio to a new file
output_path = '/Users/phonavitra/Desktop/CSIT/CSIT_DS_Mini-Challenge/CSIT_DS_Mini-Challenge/Task_1/T1_output.wav'  # Replace with your desired output path
sf.write(output_path, audio_stretched, sr)  # Replace `audio_reversed` with `audio_stretched` if using time stretch
##<OUTPUT IN RESULT.MD>##



###TASK 2###
# Breaking it down:
# 1) I will have visualize the audio (spectrogram)
# 2) Output will be a time
####################
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
file_path = '/Users/phonavitra/Desktop/CSIT/CSIT_DS_Mini-Challenge/CSIT_DS_Mini-Challenge/Task_2/T2_audio_a.wav'  # Replace with the path to your audio file
audio, sr = librosa.load(file_path, sr=None)

# Create a spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(audio), ref=np.max),
                         sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

##<OUTPUT IN RESULT.MD>##



###TASK 3###






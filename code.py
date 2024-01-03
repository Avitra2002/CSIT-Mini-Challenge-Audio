###TASK 1###
# Breaking it down:
# 1) From the hints, I would need to reverse the sounds(maybe I need to chnage it to an array and reverse it)
# 2) basic audio maniputlations like speed up or down might need to be applied
# 3) I believe the output might give me a location
####################
import librosa
import soundfile as sf

# Load the audio file
file_path = "CSIT-Mini-Challenge-Audio/Task_1/T1_audio.wav"## Replace with the file directory with local path
audio, sr = librosa.load(file_path, sr=None)

# Reverse the audio
audio_reversed = audio[::-1]

#Adjust the speed if necessary (you can change the rate to speed up or slow down)
audio_stretched = librosa.effects.time_stretch(audio_reversed, rate=1.0)

# Save the reversed (and possibly speed-adjusted) audio to a new file
output_path = '/CSIT-Mini-Challenge-Audio/Task_1/T1_output.wav'  # Replace with your desired output path
sf.write(output_path, audio_stretched, sr)
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
#plot 1
# Load the audio file
file_path = 'CSIT-Mini-Challenge-Audio/Task_2/T2_audio_a.wav'  # Replace with the path to the audio file
audio, sr = librosa.load(file_path, sr=None)

# Create a spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(audio), ref=np.max),
                         sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

#plot 2
# Load the audio file
file_path = 'CSIT-Mini-Challenge-Audio/Task_2/T2_audio_b.wav'  # Replace with the path to the audio file
audio, sr = librosa.load(file_path, sr=None)

# Create a spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(audio), ref=np.max),
                         sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

#plot 3
# Load the audio file
file_path = 'CSIT-Mini-Challenge-Audio/Task_2/T2_audio_c.wav'  # Replace with the path to the audio file
audio, sr = librosa.load(file_path, sr=None)

# Create a spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(audio), ref=np.max),
                         sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

#plot 4
# Load the audio file
file_path = 'CSIT-Mini-Challenge-Audio/Task_2/T2_audio_d.wav'  # Replace with the path to the audio file
audio, sr = librosa.load(file_path, sr=None)

# Create a spectrogram
plt.figure(figsize=(20, 8))
librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(audio), ref=np.max),
                         sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

##<OUTPUT IN RESULT.MD>##



###TASK 3###
# Breaking it down:
# 1) Perform STFT and filter to a audio data (100-2000 Hz)
# 2) Increase the amplitude after cleaning and filtering
# 3) The output might be in foreign language (I am assuming Turkish from "kökenlerivle ilgili"), and hence I need to translate it.

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft

# Load the audio file
file_path = 'CSIT-Mini-Challenge-Audio/Task_3/C.Noisy_Voice.wav'  # Replace with the path to the audio file
sr, audio = wavfile.read(file_path)

# Perform STFT
n_fft = 2048  # You might need to tweak this depending on your specific audio file
hop_length = n_fft // 4
frequencies, times, stft_matrix = stft(audio, fs=sr, nperseg=n_fft, noverlap=hop_length)

# Define the frequency range for filtering (100 Hz to 2000 Hz)
lowcut_idx = np.where(frequencies >= 100)[0][0]
highcut_idx = np.where(frequencies <= 2000)[0][-1]

# Apply bandpass filtering in the frequency domain
stft_matrix[:lowcut_idx, :] = 0
stft_matrix[highcut_idx + 1:, :] = 0

# Perform the Inverse STFT
_, audio_filtered = istft(stft_matrix, fs=sr, nperseg=n_fft, noverlap=hop_length)

# Normalize the audio
max_value = np.max(np.abs(audio_filtered))
audio_normalized = audio_filtered / max_value

# Apply gain amplification
gain_factor = 8  # This can be adjusted, but be cautious to prevent clipping
audio_amplified = audio_normalized * gain_factor

# Ensure the amplified audio doesn't exceed [-1, 1] to prevent clipping
audio_amplified = np.clip(audio_amplified, -1, 1)

# Convert back to appropriate data type for WAV file
audio_output = np.int16(audio_amplified * 32767)

# Save the filtered and normalized audio
output_path = file_path = 'CSIT-Mini-Challenge-Audio/Task_3/task3_Output.wav'  # Replace with the path to the audio file
wavfile.write(output_path, sr, audio_output)

### Readme file for whisper: https://github.com/openai/whisper/blob/main/README.md

##Translating
import whisper
model = whisper.load_model("large-v2")
filepath= "/Users/phonavitra/Desktop/CSIT/CSIT_DS_Mini-Challenge/CSIT_DS_Mini-Challenge/Task_3/task3_Output.wav"
audio = whisper.load_audio(filepath)
audio = whisper.pad_or_trim(audio)

result = model.transcribe(audio, language="en")
print(result["text"])

##<OUTPUT IN RESULT.MD>##






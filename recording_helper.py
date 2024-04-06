import numpy as np
import pyaudio
from time import sleep
import speech_recognition
import os
import wave
import io
import tensorflow as tf
import soundfile as sf
import tensorflow_io as tfio
import sounddevice

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
p = pyaudio.PyAudio()

#'''
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    print(p.get_device_info_by_host_api_device_index(0, i).get('name'))
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
#'''

'''
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
micIndex = 0

for i in range(0, numdevices):
    if str((p.get_device_info_by_host_api_device_index(0, i).get('name'))) == 'USB Audio: - (hw:3,0)':
        micIndex = i
        print('micIndex set to: ', p.get_device_info_by_host_api_device_index(0, i).get('name'))
'''



'''
samplerates = 16000, 32000, 44100, 48000, 96000, 128000
device = 1

supported_samplerates = []
for fs in samplerates:
    try:
        sounddevice.check_output_settings(device=device, samplerate=fs)
    except Exception as e:
        print(fs, e)
    else:
        supported_samplerates.append(fs)
print(supported_samplerates)
'''

def record_audio(index):
    stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
            input_device_index=index
        )

    #print("start recording...")

    frames = []
    seconds = 1
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)
        #print(frames)
        
    #print("recording stopped")

    stream.stop_stream()
    stream.close()

    #return np.frombuffer(b''.join(frames), dtype=np.int16)
    #return b''.join(frames)

#'''
    obj = wave.open('tmp.wav', 'wb')
    obj.setnchannels(CHANNELS)
    obj.setsampwidth(p.get_sample_size(FORMAT))
    obj.setframerate(RATE)
    obj.writeframes(b''.join(frames))
    obj.close
    
    return 'tmp.wav'
#'''
'''
    container = io.BytesIO()
    wf = wave.open(container, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close

    container.seek(0)
    data_package = container.read()
    return data_package
'''

def terminate():
    p.terminate()


'''
class Recorder():

    sampling_rate = 16000
    num_channels = 2
    sample_width = 4 # The width of each sample in bytes. Each group of ``sample_width`` bytes represents a single audio sample. 

    def pyaudio_stream_callback(self, in_data, frame_count, time_info, status):
        self.raw_audio_bytes_array.extend(in_data)
        return (in_data, pyaudio.paContinue)

    def start_recording(self):

        self.raw_audio_bytes_array = bytearray()

        pa = pyaudio.PyAudio()
        self.pyaudio_stream = pa.open(format=pyaudio.paInt16,
                                      channels=self.num_channels,
                                      rate=self.sampling_rate,
                                      input=True,
                                      stream_callback=self.pyaudio_stream_callback)

        self.pyaudio_stream.start_stream()

    def stop_recording(self):

        self.pyaudio_stream.stop_stream()
        self.pyaudio_stream.close()

        speech_recognition_audio_data = speech_recognition.AudioData(self.raw_audio_bytes_array,
                                                                     self.sampling_rate,
                                                                     self.sample_width)
        return speech_recognition_audio_data


if __name__ == '__main__':

    recorder = Recorder()

    # start recording
    recorder.start_recording()

    # say something interesting...
    sleep(3)

    # stop recording
    speech_recognition_audio_data = recorder.stop_recording()

    # convert the audio represented by the ``AudioData`` object to
    # a byte string representing the contents of a WAV file
    wav_data = speech_recognition_audio_data.get_wav_data()
'''

'''
def record_audio():
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    frames = []
    seconds = 1
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    return np.frombuffer(b''.join(frames), dtype=np.int16)

def terminate():
    p.terminate()
'''

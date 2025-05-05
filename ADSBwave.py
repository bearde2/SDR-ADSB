import sys
sys.path.append('wave-bwf-rf64/src/')
import wave_bwf_rf64 as wave
import os
import matplotlib.pyplot as plt

source = os.path.join('23-38-54_1090100000Hz','23-38-54_1090100000Hz.wav')
print(source)

# with wave.open(source) as file:
# 	print(file)
a = wave.open(source)
framerate = a.getframerate()
frames = a.readframes(a.getnframes())

import struct
format_string = "<" + "h" * (len(frames) // 2)
pcm_samples = struct.unpack(format_string, frames)
len(pcm_samples)

import numpy as np
pcm_samples = np.frombuffer(frames, dtype="<h")
normalized_amplitudes = abs(pcm_samples / (2 ** 15))
time = np.array(np.arange(0,len(pcm_samples))/framerate)#s
timeus = time/1e-6 #us

plt.figure()
plt.plot(timeus,np.transpose(normalized_amplitudes))

digital_threshold = .6
digital_norm_pcm = np.where(np.transpose(normalized_amplitudes)>=digital_threshold,1,0)
digital_norm_pcm = digital_norm_pcm[3500000:4700000]


preamble = np.array([-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1]) #15 from first 0
preamble_conv = np.convolve(digital_norm_pcm,preamble)
# preamble_conv = np.convolve(np.transpose(normalized_amplitudes),preamble)
preamble_conv = np.where(preamble_conv[15:-19]>1,1,0)

plt.figure()
# plt.plot(np.transpose(normalized_amplitudes))
plt.plot(preamble_conv)
plt.plot(digital_norm_pcm,'-o')
# plt.show()
x=1
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

import numpy as np
pcm_samples = np.frombuffer(frames, dtype="<h")
normalized_amplitudes = (pcm_samples / (2 ** 15))
time = np.array(np.arange(0,len(pcm_samples))/framerate)#s


# plt.figure()
# plt.plot(timeus,np.transpose(normalized_amplitudes))

###****** interpolate to 0.25us timesteps
import scipy.interpolate as sp
# interp_timestep = 1/(5*2.4e6) # 1/5 acq s/samp
interp_timestep = 5e-7 # .5us/samp
timeint = np.arange(time[0],time[-1],interp_timestep)
normalized_amplitudesint = np.interp(timeint,time,np.transpose(normalized_amplitudes))
# normalized_amplitudesint_obj = sp.CubicSpline(time,np.transpose(normalized_amplitudes))
# normalized_amplitudesint_obj = sp.PchipInterpolator(time,np.transpose(normalized_amplitudes))
# normalized_amplitudesint = normalized_amplitudesint_obj(timeint)
###******


timeus = time/1e-6 #us
timeusint = timeint/1e-6 #us

digital_threshold = .6
digital_thresholdint = .6
digital_norm_pcm = np.where(np.transpose(normalized_amplitudes)>=digital_threshold,1,0)
digital_norm_pcmint = np.where(normalized_amplitudesint>=digital_thresholdint,1,0)

ind1 = 3770000; ind2 = 3800000
ind1 = 3670000; ind2 = 3700000
# ind1 = 0
# ind2 = len(digital_norm_pcm)
ind1int = round(ind1*len(timeint)/len(time))
ind2int = round(ind2*len(timeint)/len(time))

digital_norm_pcm = digital_norm_pcm[ind1:ind2]
digital_norm_pcmint = digital_norm_pcmint[ind1int:ind2int]
timeus = timeus[ind1:ind2]
timeusint = timeusint[ind1int:ind2int]

preamble = np.array([-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1]) #15 from first 0
preambleint = np.array([-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, -1,-1,-1,-1,-1,1, 1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1, 1, -1,-1,-1,-1,-1,1, 1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]) #.25us interp
preamble_conv = np.convolve(digital_norm_pcm,preamble)
preamble_convint = np.convolve(digital_norm_pcmint,preambleint)

preamble_conv = np.where(preamble_conv[15:-19]>1,1,0)
preamble_convint = np.where(preamble_convint[45:-29]>1,1,0) # .25us interp



# digital_norm_pcm = signal.resample(digital_norm_pcm, signalStepsInDurationR)
# timeusNew = np.linspace(timeus[0],timeus[-1],signalStepsInDurationR, endpoint=False)

preamble_indices = np.where(preamble_conv==1)[0]
# preamble_indices = np.where(preamble_convint==1)[0]
packet_raw = np.zeros((len(preamble_indices),2*112))
packet_bits = np.zeros((len(preamble_indices),104))
for i in range(len(preamble_indices)):
	packet_raw[i,:] = digital_norm_pcm[preamble_indices[i]:preamble_indices[i]+4*112:2]
	# packet_raw[i,:] = digital_norm_pcmint[preamble_indices[i]:preamble_indices[i]+4*112:2]
	count = 0
	plt.figure()
	plt.plot(digital_norm_pcm[preamble_indices[i]:preamble_indices[i]+8*112:2])
	plt.show()
	# for j in range(15,2*112-1,2):
	# 	if packet_raw[i,j]-packet_raw[i,j+1]>0:
	# 		packet_bits[i,count] = 1
	# 		count+=1
	# 	else:
	# 		packet_bits[i,count] = 0
	# 		count+=1

plt.figure()
plt.plot(timeus,digital_norm_pcm,'-o')
plt.plot(timeusint,digital_norm_pcmint,'-o',mfc='none')
plt.plot(timeus,preamble_conv)
plt.plot(timeusint,preamble_convint)
plt.legend(['norm','int','pream','preamint'])
x=1
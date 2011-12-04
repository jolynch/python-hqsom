'''
Module for importing audio and computing spectrograms to be fed to hierarchies
as list-of-lists representations.

Think of this as analogous to what it would take to convert a video into
something that one could feed to a hierarchy, except for sound instead.

This uses only mono WAV files.  Convert your mp3 files using some external
program first if you want to use them with this.
'''

import wave
import numpy as np
import struct



# Frequency-domain representations of audio, from using a windowed FFT on
# successive intervals of a time-domain representation of the file.
# Essentially creates a list of arrays, each of which is an estimate
# of the spectral density over an interval.
class Spectrogram(object):
    
    # imports a given mono WAV file
    def __init__(self, filename):
        self.wavfile = wave.open(filename, 'rb')
        self.sample_rate = self.wavfile.getframerate()
        self.total_num_samples = self.wavfile.getnframes()
    
    # FFT_length should be a power of 2 for best efficiency
    def getSpectrogram(self, FFT_length):
        num_fft_blocks = (self.total_num_samples / FFT_length) - 2
        
        # unpacked representation of a block of the wav file
        temp = np.zeros((num_fft_blocks,FFT_length),dtype=np.float64)
        
        # read in the data from the file
        for i in range(num_fft_blocks):
            tempb = self.wavfile.readframes(FFT_length);
            temp[i,:] = np.array(struct.unpack("%dB"%(FFT_length), \
                        tempb),dtype=np.float64) - 128.0
        self.wavfile.close()
        
        # window the data
        temp = temp * np.hamming(FFT_length)
        
        # Transform with the FFT, Return Power
        freq_pwr  = 10*np.log10(1e-20+np.abs(np.fft.rfft(temp,FFT_length)))
        
        # Plot the result
        n_out_pts = (FFT_length / 2) + 1
        y_axis = 0.5*float(self.sample_rate) / n_out_pts * np.arange(n_out_pts)
        x_axis = (self.total_num_samps / float(self.sample_rate)) / \
                 num_fft_blocks * np.arange(num_fft_blocks)




'''
Refs:

http://stackoverflow.com/questions/1303307/fft-for-spectrograms-in-python

THIS ONE IS THE BEST:
http://macdevcenter.com/pub/a/python/2001/01/31/numerically.html?page=1
http://macdevcenter.com/pub/a/python/2001/01/31/numerically.html?page=2
http://onlamp.com/python/2001/01/31/graphics/pysono.py

LIBRARIES WE NEED FOR IT:
http://docs.python.org/library/wave.html
http://docs.scipy.org/doc/numpy/reference/routines.fft.html

MATPLOTLIB VERSION, BUT ONLY FOR PLOTTING:
http://matplotlib.sourceforge.net/api/pyplot_api.html#matplotlib.pyplot.specgram

DIFFERENT WINDOWING OPTIONS:
http://en.wikipedia.org/wiki/Window_function#Hamming_window


'''
        
        
        
        

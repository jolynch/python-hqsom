'''
Module for importing audio and computing spectrograms to be fed to hierarchies
as list-of-lists representations.

Think of this as analogous to what it would take to convert a video into
something that one could feed to a hierarchy, except for sound instead.
'''



# Time-domain representations of audio, as super-long arrays of floats.
class timeDomain(object):
    
    def __init__(self, filename):
        # calls appropriate import method to get time-domain repr of
        # input signal
        pass

    # will rely on external library
    def import_from_wav(self, filename):
        pass

    # will rely on external library
    def import_from_mp3(self, filename):
        pass



# Frequency-domain representations of audio, built from using an FFT version of
# the short-term Fourier transform algorithm on a time-domain representation
# of an audio signal, over successive portions of the signal using a Hann
# window.  Essentially creates a list of arrays, each of which is an estimate
# of the spectral density at that point in time.
class spectrogram(object):
    def __init__(self, timeDomainSignal):
        pass
    # Runs STFT algorithm on successive chunks of the signal
    def stft_hann_window(self, intervalSize, offset, hann=True):
        pass

    # Hann windowing mitigates aliasing
    def hann_window(self, signalChunk):
        pass



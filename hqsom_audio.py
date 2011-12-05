'''
Networks for audio classification

As a goal, we want to train a network to classify sections from
particular songs.

Procedure:

1. Grab three songs; isolate n 1-second snippets from each; convert to WAV.
    i. ideally choose n even.

2. Arbitrarily select n-3 snippets from each song to be in-sample data.

3. Select a network from one of the below designs.

4. Train the network on the in-sample data using the audio train/test framework.
    i. choose a song
    ii. choose a spectrogram FFT window size, ideally pretty long
    iii. generate spectrograms for each of the in-sample snippets
    iv. construct a fully-connected graph between the snippets as nodes
    v. generate an Eulerian path on the graph
        a. this is why we wanted n to be even
        b. then n-3 is odd so an Eulerian path exists
    vi. cycle through the Eulerian path in forward and reverse order to train
                the network for this specific song
    vii. if your top-level node is not constant, you're doing it wrong
    viii. feed the network a long period of silence
    ix. if your top-level node did not change to a different constant, you're
                likewise doing it wrong
    x. choose the next song; GOTO (i)

5. Test classification performance
    i. choose a song
    ii. generate spectrograms of the out-of-sample snippets using the
                same window size as in training above
    iii. feed in each spectrogram and see if the top-level node produces the
                same output as it did during training for this song

'''









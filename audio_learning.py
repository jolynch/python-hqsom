import random
import preproc.audio as audio
import getopt, sys
import traceback
import pickle
import numpy as np
import timeit
import sys, traceback

def score_audio(genome, setup_data=None):
    try:
        final_data = setup_data["data"]
        song_types = setup_data["types"]
        output_size = setup_data["output_size"]
        clusters = setup_data["clusters"]
        timeout = setup_data["timeout"]

        hqsom = genome.to_hierarchy()

        # Slow networks are genetically unfit
        mock_data = [.1] * genome.input_size
        single_update = timeit.timeit(lambda: hqsom.update(mock_data), number=1)
        if single_update > timeout:
            return -50

        # Testing schema:
        # 1) Expose to entirety of three songs
        # 2) Pick 3 random sequences of test_length in size from each song, run through
        # 3) Clear at each in between
        num_seconds, test_length  = .1, 10
        num_cycles, num_repeats = 1, 1
        for i in range(num_cycles):
            for song_type in song_types:
                if song_type == "ClassicalTEST" or song_type == "TechnoTEST" or song_type == "RockTEST":
                    continue
                for spectrum in final_data[song_type]:
                    hqsom.update(spectrum)
                hqsom.reset()

        for i in range(num_cycles*2):
            for song_type in song_types:
                if song_type == "ClassicalTEST" or song_type == "TechnoTEST" or song_type == "RockTEST":
                    continue
                num_spectrograms = len(final_data[song_type])
                r_index = np.random.randint(0,num_spectrograms-test_length)
                for index in range(r_index, r_index+test_length):
                    hqsom.update(final_data[song_type][index])
                hqsom.reset()

        # Done with testing, gather results
        all_results = {}
        for data_name in song_types:
            data_collection = final_data[data_name]
            results =[0]*output_size
            for spect in data_collection:
                results[hqsom.activation_vector(spect)] += 1
            t = sum(results)
            results = [float(i)/t for i in results]
            results = np.array(results)
            all_results[data_name] = results

        # Purity metric, we like overfitting because we're trying to encompass a
        # lot of noise tolerance, so if we can make big clusters that separate the
        # data, cool
        purity = [0] * output_size
        def score_column(column, clusters, results):
            cluster_scores = {}
            for cluster in clusters:
                cluster_score = 0
                for name in cluster:
                    cluster_score += results[name][column]
                cluster_scores[cluster] = cluster_score
            cluster_scores = cluster_scores.items()
            scores = [score for (i, score) in cluster_scores]
            golden = np.argmax(scores)
            return 2 * scores[golden] - np.sum(scores)

        for i in range(output_size):
            purity[i] = score_column(i, clusters, all_results)

        return np.sum(purity)

    except Exception as e:
        print "Tester failed, error, could be related to genome"
        print str(e)
        print genome
        traceback.print_exc(file=sys.stdout)
        return -100

def setup_audio():
    print "Loading songs into memory"
    song_rock = audio.Spectrogram("data/music/Californication.wav")
    song_rock2 = audio.Spectrogram("data/music/ByWay.wav")
    song_techno = audio.Spectrogram("data/music/Everybody.wav")
    song_techno2 = audio.Spectrogram("data/music/DayNNight.wav")
    song_classical = audio.Spectrogram("data/music/Bells.wav")
    song_classical2 = audio.Spectrogram("data/music/Symp9.wav")

    print "Done loading songs into memory"
    songs = [
            ("Techno", song_techno), 
            ("TechnoTEST", song_techno2),
            ("Classical", song_classical),
            ("ClassicalTEST", song_classical2),
            ("Rock", song_rock), 
            ("RockTEST", song_rock2),
    ]
    clusters = [('Classical', 'ClassicalTEST'), ('Rock', 'RockTEST'), ('Techno', 'TechnoTEST')]
    song_types = [i for (i,j) in songs]
    num_seconds, test_length  = .1, 10
    #Get num_second second slices of each song, looking to a cache first
    try:
        (n,saved_songs,final_data) = pickle.load(open("cache.p", "rb"))
        if not n == num_seconds or not saved_songs == tuple(song_types):
            raise Exception
        print "Found data in cache, skipping generation"
    except:
        print "Generating ffts"
        raw_data = dict([(i,None) for i in song_types])
        for (song_type, song_file) in songs:
            print "Generating data on the fly for {} song".format(song_type)
            fft_length = song_file.sample_rate * num_seconds
            #To get a power of 2
            fft_length = int(2**np.ceil(np.log(fft_length)/np.log(2)));
            print "Using fft_length of {}".format(fft_length)
            raw_data[song_type] = song_file.get_spectrogram(fft_length)

        print "Reshaping ffts into length 128 inputs"
        final_data = {}
        for song_type in song_types:
            data = raw_data[song_type]
            new_data = np.zeros((data.shape[0], 128))
            bucket_sum, spect = 0, None
            for spect_index in range(len(data)):
                print "{} of {} Spectrograms processed".format(spect_index, len(data))
                spect = data[spect_index]
                window_size = len(spect) / 128
                bucket_sum = 0
                for i in range(128):
                    new_data[spect_index][i] = spect[i*window_size]

            final_data[song_type] = new_data 
        pickle.dump((num_seconds, tuple(song_types), final_data), open("cache.p","wb"))
    return song_types, final_data, clusters


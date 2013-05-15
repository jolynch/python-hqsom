import math
import random
import pprint
from genetic_algo import *
from multiprocessing import Pool, cpu_count
from functools import partial
from copy import deepcopy
import preproc.audio as audio
import getopt, sys
import traceback
import pickle
import numpy as np
import timeit
import sys, traceback

def exponential_choice(array, idx=True):
    """ Random choice that prefers earlier elements """
    index = int(np.random.standard_exponential()) % len(array)
    if idx:
        return index
    else:
        return array[index]

def GenerateGeneration(input_size, output_size, size=80, seed=None):
    next_generation = []
    if seed is None:
        return [Genome(input_size, output_size) for i in range(size)]
    else:
        #assume that seed is of the form
        # [(genome, score)...]
        # and that it is N long, goal is one of size M
        sorted_genes = sorted(seed, key=lambda x: -x[1])
        # Get N/2 genes to work with
        better_half = sorted_genes[:len(sorted_genes)/4]
        resulting_generation = []

        resulting_generation.append(better_half[0][0])
        resulting_generation.append(better_half[1][0])

        # Fill in N/4 of the results with recombinations
        while len(better_half) > 2:
            left = better_half.pop(exponential_choice(better_half))[0]
            right = better_half.pop(exponential_choice(better_half))[0]
            print "Crossing "
            pprint.pprint(left)
            print " with "
            pprint.pprint(right)
            resulting_generation.append(left.combine(right))
            print "Got"
            pprint.pprint(resulting_generation[-1])
            resulting_generation.append(right.combine(left))

        # Fill in N/4 of the results with mutations
        better_half = sorted_genes[:len(sorted_genes)/4]
        for genome_score in better_half:
            genome = genome_score[0]
            resulting_generation.append(genome.mutate(.05, .2, .5))

        for genome_score in better_half:
            genome = genome_score[0]
            resulting_generation.append(genome.mutate(.4, .05, .05))


        while len(resulting_generation) < size:
            resulting_generation.append(Genome(input_size, output_size))

        return resulting_generation[:size]


def score_audio(genome, setup_data=None):
    try:
        final_data = setup_data["data"]
        song_types = setup_data["types"]
        output_size = setup_data["output_size"]
        clusters = setup_data["clusters"]
        timeout = setup_data["timeout"]

        num_seconds, test_length  = .1, 10

        hqsom = genome.to_hierarchy()

        mock_data = [.1] * genome.input_size
        single_update = timeit.timeit(lambda: hqsom.update(mock_data), number=1)
        if single_update > timeout:
            return -50
        #Testing schema:
        # 1) Expose to entirety of three songs
        # 2) Pick 3 random sequences of test_length in size from each song, run through
        # 3) Clear at each in between
        seq_num = 0
        num_cycles, num_repeats = 1, 1
        total_run_count = num_cycles*sum([(len(final_data[x])) for x in song_types])

        for i in range(num_cycles):
            for song_type in song_types:
                if song_type == "ClassicalTEST" or song_type == "TechnoTEST" or song_type == "RockTEST":
                    continue
                for spectrum in final_data[song_type]:
                    hqsom.update(spectrum)
                    #print hqsom.activation_vector(spectrum, True, True)
                    seq_num += 1
                hqsom.reset()

        total_run_count = num_cycles*2*len(song_types)*test_length
        seq_num = 0
        for i in range(num_cycles*2):
            for song_type in song_types:
                if song_type == "ClassicalTEST" or song_type == "TechnoTEST" or song_type == "RockTEST":
                    continue
                num_spectrograms = len(final_data[song_type])
                r_index = np.random.randint(0,num_spectrograms-test_length)
                for index in range(r_index, r_index+test_length):
                    hqsom.update(final_data[song_type][index])
                    #print hqsom.activation_vector(spectrum, False, True)
                    seq_num += 1
                hqsom.reset()
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
                    #bucket_sum = np.mean(spect[i*window_size:i*window_size+window_size])
                    new_data[spect_index][i] = spect[i*window_size]
                #new_data[spect_index] = new_data[spect_index] - min(new_data[spect_index])
                #new_data[spect_index] = new_data[spect_index] / np.linalg.norm(new_data[spect_index])

            final_data[song_type] = new_data 
        pickle.dump((num_seconds, tuple(song_types), final_data), open("cache.p","wb"))
    return song_types, final_data, clusters

def ScoreGeneration(generation, setup_function, score_function, name="data.p"):
    pool = Pool(processes=cpu_count())
    types, data, clusters = setup_function()
    smallest_output = min([genome.output_size for genome in generation])
    ddata = {
        "data": data,
        "types":types,
        "output_size":smallest_output,
        "clusters": clusters,
        "timeout": 1.0
    }
    scoring_function = partial(score_function, setup_data=deepcopy(ddata))
    scores = pool.map(scoring_function, generation)
    gen_data = [(generation[i], scores[i]) for i in range(len(generation))]
    # Dump it to disk
    pickle.dump(gen_data, open(name, "wb"))
    return gen_data


def RunGeneticAlgorithm(input_size,
                        output_size,
                        setup_function,
                        score_function,
                        generation_name="",
                        generation_size=80,
                        iterations=1):
    """
    Run the genetic algorithm on networks giving input_size and outputing
    output_size using setup_function and score_function as the fitness function,
    starting with generation_name (if None make a random one)
    """
    seed = None
    try:
        seed = pickle.load(open(generation_name+".p", "rb"))
    except:
        print "Could not open %s" % (generation_name+".p",)
        pass

    if generation_name is None:
        generation_name = "default"

    generation = GenerateGeneration(input_size,
                                    output_size,
                                    generation_size,
                                    seed)
    gen_score = None

    for i in range(iterations):
        gen_score = ScoreGeneration(generation, setup_function, score_function,
                                    name="%s-%d.p"%(generation_name, i))
        generation = GenerateGeneration(input_size, output_size,
                                        generation_size, gen_score)

    return gen_score




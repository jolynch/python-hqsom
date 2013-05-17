import math
import random
import pprint
from genetic_algo import *
from multiprocessing import Pool, cpu_count
from functools import partial
from copy import deepcopy
from letter_learning import *
from audio_learning import * 
import getopt, sys
import traceback
import pickle
import numpy as np
import sys, traceback

def exponential_choice(array, idx=True):
    """ Random choice that prefers earlier elements """
    index = int(np.random.standard_exponential()) % len(array)
    if idx:
        return index
    else:
        return array[index]

def GenerateGeneration(input_size, output_size, size=80, seed=None):
    """
    If no seed generation is provided, generate a completely random guess.

    Otherwise, do intelligent combinations and mutations of the existing seed
    dataset to generate 3/4 of the dataset, and fill in the last 1/4 with
    random genomes to prevent local maxima
    """

    next_generation = []
    if seed is None:
        return [Genome(input_size, output_size) for i in range(size)]
    else:
        #assume that seed is of the form
        # [(genome, score)...]
        # and that it is N long, goal is one of size M
        sorted_genes = sorted(seed, key=lambda x: -x[1])

        # Get N/4 genes to work with
        better_half = sorted_genes[:len(sorted_genes)/4]
        resulting_generation = []

        # Always keep the top two, the next generation ought not be worse than
        # this one.
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

        # Fill in N/4 of the results with structural mutations
        better_half = sorted_genes[:len(sorted_genes)/4]
        for genome_score in better_half:
            genome = genome_score[0]
            resulting_generation.append(genome.mutate(.05, .5, .5))

        # File in N/4 of the results with parameter mutations
        for genome_score in better_half:
            genome = genome_score[0]
            resulting_generation.append(genome.mutate(.4, .05, .05))

        # Fill in the rest with random canditates
        while len(resulting_generation) < size:
            resulting_generation.append(Genome(input_size, output_size))

        return resulting_generation[:size]

def ScoreGeneration(generation, setup_function, score_function, name="data.p"):
    """
    A sort of poor man's map reduce.  First we prepare the dataset with the
    setup_function, and then we in parallel apply the score function to that
    data passing in a single genome at a time (running cpu_count() such scoring
    functions in parrallel)
    """

    pool = Pool(processes=cpu_count())
    types, data, clusters = setup_function()
    smallest_output = min([genome.output_size for genome in generation])
    ddata = {
        "data": data,
        "types":types,
        "output_size": smallest_output,
        "clusters": clusters,
        "timeout": 0.5
    }

    scoring_function = partial(score_function, setup_data=deepcopy(ddata))
    scores = pool.map(scoring_function, generation)
    gen_data = [(generation[i], scores[i]) for i in range(len(generation))]

    # Dump it to disk, this way we can always resume, as well as getting global
    # maximum later
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
        print "Scoring %d generation of size %d" % (i+1, len(generation))
        gen_score = ScoreGeneration(generation, setup_function, score_function,
                                    name="%s-%d.p"%(generation_name, i))
        generation = GenerateGeneration(input_size, output_size,
                                        generation_size, gen_score)

    return gen_score




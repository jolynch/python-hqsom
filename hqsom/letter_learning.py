import numpy as np
from PIL import Image
import sys, traceback
import timeit


class Letter(object):
    """
    Convenience class to read in the letter images and convert them
    to useful numpy arrays
    """
    def __init__(self, letter, noise=None):
        image_loc = "data/alphabet/%s.png" % letter.capitalize()
        im = Image.open(image_loc)
        # Trim the data
        self.data = np.array(im.convert("L")) / 255.0
        self.data = np.compress([False] * 4 + [True] * 8 + [False] * 4,
                                self.data, axis=1)
        self.data = np.compress([False] * 1 + [True] * 8 + [False] * 1,
                                self.data, axis=0)
        self.data = self.data.reshape(1, 64)[0]

        if noise is not None:
            noise_array = np.random.normal(0.0, noise, self.data.shape)
            self.data += noise_array

    def __getitem__(self, index):
        return self.data[index]


def score_letters(genome, setup_data=None):
    """
    We want to form the concept of a letter that is noise tolerant, so we show the
    network lots of A's, B's, C's and D's and then ask it to cluster noisy
    variants.  A basic purity metric is used as the scoring function
    """
    try:
        data = setup_data["data"]
        letters = setup_data["types"]
        output_size = genome.output_size
        clusters = setup_data["clusters"]
        timeout = setup_data["timeout"]

        hqsom = genome.to_hierarchy()
        total_repeat = 5
        letter_repeat = 25

        # Slow genomes are unfit
        mock_data = [.1] * genome.input_size
        single_update = timeit.timeit(lambda: hqsom.update(mock_data), number=1)
        if single_update > timeout:
            return -50

        # Testing Strategy:
        # Expose the network to the clean letter many times, then reset the
        # RSOM difference matrices and do the same with the next letter.
        # Do this total_repeat times, and that's all she wrote

        for repeat in range(total_repeat):
            for letter in letters:
                for i in range(letter_repeat):
                    hqsom.update(data["clean-%s" % letter])
                hqsom.reset()

        all_results = {}
        all_labels = [i for j in clusters for i in j]
        for label in all_labels:
            all_results[label] = hqsom.activation_vector(data[label],
                                                         continuous_output=True)

        # Purity metric, we like overfitting because we're trying to encompass a
        # lot of noise tolerance, so if we can make big clusters that separate the
        # data, cool
        purity = [0] * genome.output_size
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

        for cluster in clusters:
            print cluster[0], [hqsom.activation_vector(data[c]) for c in cluster]

        return np.sum(purity)

    except Exception as e:
        print "Tester failed, could be related to genome"
        print str(e)
        print genome
        traceback.print_exc(file=sys.stdout)
        return -100

def setup_letters():
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    data = {}
    clusters = []
    for l in letters:
        data["clean-%s" % l] = Letter(l).data

    np.random.seed(15717)
    for l in letters:
        noisy_letters = []
        for i in range(1):
            data["noisy-%s-%d" % (l, i)] = Letter(l, .1).data
            noisy_letters.append("noisy-%s-%d" % (l, i))
        clusters.append(tuple(["clean-%s" % l] + noisy_letters))

    return [l for l in letters], data, clusters

## "letter_first" output size 5
## Testing setup for the ABCD separation
## Noise of .1 stddev, 5 repeats of 25 exposure per letter
## 5 data points, 1 clean data point for training => letter_first

## "twelve" output size 16
## Testing setup for the ABCD EFGH IJKL separation
## Noise of .25 stddev, 3 repeats of 25 exposure per letter
## 10 data points, 1 clean data point for training

## high_noise output size 5
## Testing setup for the ABCD separation
## Noise of .3 stdev, 5 repeats of 25 exposure per letter
## 10 data points, 1 clean data point for training

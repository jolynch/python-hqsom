import math
import random
import pprint
from hqsom_audio import *

"""
Each genome is variable length in the number of LayerConf1D genes, going from
the lowest levels on the left to the highest levels on the right

Each genome has the form:
[
    initial_input_size  -  Must be a factor of 2
    output_size - number of clusters we are looking for
    [gene1, gene2, gene3 ...] - List of genes that represent the layers
]

Each gene has the following layout (see LayerConf1D for details):
[
    size            -  Number of hqsom nodes in this layer
    inputs          -  Number of total inputs to this layer
    overlap_pct     -  Percent of inputs that should overlap

    m_s             -  SOM map size for each node
    gamma_s         -  SOM gamma
    sigma_s         -  SOM sigma

    m_r             -  RSOM map size
    alpha_r         -  RSOM alpha value
    gamma_r         -  RSOM gamma
    sigma_r         -  RSOM sigma
]

Each genome is a list of these genes, at any time the mutations can either
change the number of genes or change the parameters of the associated layers
"""

# Stolen from stack overflow ... completely
def factors(n):
    return list(set(reduce(list.__add__,
                      ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

class Gene(object):
    def __repr__(self):
        return str(self.data)

    """
    A gene in this system.
    """
    def __init__(self, num_inputs, num_output, data=None):
        if data is not None:
            self.data = data
        else:
            self.data = [
                num_output,
                num_inputs,
                random.random(),

                random.randint(10, 200),
                random.random(),
                random.randint(2, 500),

                random.randint(num_output + 1, 200),
                random.random(),
                random.random(),
                random.randint(2, 500)
            ]

    def to_config(self):
        data = self.data
        return LayerConf1D(data[0],
                           data[1] / data[0],
                           data[1],
                           0,
                           data[3],
                           data[4],
                           data[5],
                           data[6],
                           data[7],
                           data[8],
                           data[9])


class Genome(object):
    def __repr__(self):
        return pprint.pformat([(self.input_size, self.output_size)] + self.genes)

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        num_layers = random.randint(1, 10)
        """
        # This doesn't work
        def lfunc(x):
            base = ((output_size/float(input_size))**(1.0/num_layers))
            return int(math.ceil(input_size * math.pow(base, x)))
        """
        size = input_size
        self.genes = []
        while True:
            next_size = random.choice(factors(size))
            self.genes.append(Gene(size, next_size))
            if next_size == 1:
                break
            else:
                size = next_size
        self.genes[-1].data[6] = output_size


    def to_hierarchy(self):
        layer_configs = [gene.to_config() for gene in self.genes]
        return apply(Hierarchy1D, layer_configs)

    def combine(self, other):
        left, right = self.data, other.data
        return other


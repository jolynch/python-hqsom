import math
import random
import pprint
from copy import deepcopy
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

def test(prob):
    return random.random() < prob

# Stolen from stack overflGow ... completely
def factors(n):
    return list(set(reduce(list.__add__,
                      ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

roundable = [True, True, False, True, False, True, True, False, False, True]

class Gene(object):
    def __repr__(self):
        return str(self.data)

    """
    A gene in this system.
    """
    def __init__(self, num_inputs, num_output, data=None):
        if data is not None:
            self.data = deepcopy(data)
            self.data[0] = num_inputs
            self.data[1] = num_output
        else:
            self.data = [
                num_inputs,
                num_output,
                random.random(),

                random.randint(10, 100),
                random.random(),
                random.randint(2, 500),

                random.randint(num_output + 1, num_output + 100),
                random.random(),
                random.random(),
                random.randint(2, 500)
            ]

    def input(self):
        return self.data[0]

    def output(self):
        return self.data[1]

    def mutate(self, prob=.1):
        """
        Returns a mutated gene, does not modify self
        """
        gene_data = deepcopy(self.data)
        alternative = Gene(self.data[0], self.data[1])
        for i in range(2, len(self.data)):
            if test(prob):
                gene_data[i] = alternative.data[i]
        return Gene(self.data[0], self.data[1], gene_data)

    def combine(self, other):
        """
        Combines this gene with the other, basically averaging the two
        parameter sets and using self.input_size and other.output_size.
        """
        gene_data = deepcopy(self.data)
        gene_data[1] = other.data[1]
        for i in range(2, len(gene_data)):
            gene_data[i] = (gene_data[i] + other.data[i]) / 2
            if roundable[i]:
                gene_data[i] = int(round(gene_data[i]))

        return Gene(gene_data[0], gene_data[1], gene_data)

    def to_config(self):
        data = self.data
        return LayerConf1D(data[1],
                           data[0] / data[1],
                           data[0],
                           0,
                           data[3],
                           data[4],
                           data[5],
                           data[6],
                           data[7],
                           data[8],
                           data[9])

class Genome(object):
    """
    Class to represent a Genome that can be used to generate a HQSOM Layered
    Network
    """
    def __repr__(self):
        return pprint.pformat([(self.input_size, self.output_size)] + self.genes)

    def __init__(self, input_size, output_size, genes=None):
        self.input_size = input_size
        self.output_size = output_size

        if genes is None:
            num_layers = random.randint(1, 10)
            size = input_size
            self.genes = []
            while True:
                f = factors(size)
                next_size = random.choice(f)
                if next_size <= size:
                    self.genes.append(Gene(size, next_size))
                if len(f) <= 1:
                # Down to just [1]
                    break
                else:
                    size = next_size
            # Output size of the last som should just be the overall size
            self.genes[-1].data[6] = output_size
        else:
            self.genes = deepcopy(genes)


    def to_hierarchy(self):
        layer_configs = [gene.to_config() for gene in self.genes]
        return apply(Hierarchy1D, layer_configs)

    def combine(self, other, crossover_probability=1):
        """
        Does simple crossover recombinations, a crossover point is selected
        between the left and right genome, to the left of that point we use
        genes from left, to the right we use from right 
        """
        left, right = deepcopy(self.genes), deepcopy(other.genes)
        new_genes = []

        if test(crossover_probability):
            xpoint = random.randint(0, len(left)-1)
            for i in range(xpoint):
                new_genes.append(left[i])
            rpoint = 0
            while(right[rpoint].output() > left[xpoint].output()):
                rpoint = rpoint + 1

            new_genes.append(left[xpoint].combine(right[rpoint]))
            for i in range(rpoint+1, len(right)):
                new_genes.append(right[i])

            # Output size of the last som should just be the overall size
            new_genes[-1].data[6] = self.output_size

        return Genome(self.input_size, self.output_size, new_genes)



    def mutate(self, prob=.1, prob_split=.05, prob_join=.05):
        """
        Mutate this genome, does three possible mutations:
        1) Mutates the genes with some probability
        2) Splits a gene into two genes, with the new gene being random
        3) Joins two genes into one gene

        Returns a genome that has been mutated, does not mutate genome state
        """

        # With some probability mutate genes in this genome
        new_genes = []
        for gene in self.genes:
            new_genes.append(gene.mutate(prob))

        # With some probability split a gene
        if test(prob_split):
            splits = {}
            for gene in new_genes:
                gfactors = factors(gene.input())
                potential_split = [f for f in gfactors
                                   if f > gene.output() and
                                   f < gene.input()]
                if potential_split:
                    chosen = random.choice(potential_split)
                    splits[gene] = chosen

            split_point = random.choice(splits.keys())
            index = new_genes.index(split_point)
            new_genes.insert(index + 1,Gene(splits[split_point], split_point.output()))
            new_genes[index].data[1] = splits[split_point]

        # With some probability join a gene
        if test(prob_join):
            joins = {}
            for gene in new_genes[:-1]:
                joins[gene] = True
            if joins:
                join_point = random.choice(joins.keys())
                index = new_genes.index(join_point)
                new_genes[index] = new_genes[index].combine(new_genes[index+1])
                new_genes.pop(index+1)

        # Output size of the last som should just be the overall size
        new_genes[-1].data[6] = self.output_size

        return Genome(self.input_size, self.output_size, new_genes)


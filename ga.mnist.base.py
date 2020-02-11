import numpy as np

import cga

import pickling_ga
from search_space_reimpl import search_space


folder = 'experiments/'

# Set up evaluation
population_size = 20

epochs = 6


def const_epochs_6(individual, gen_index, prev_evaluations):
    return epochs


# original paper
learning_rate_decay_points = [1, 149, 249]

relative_decay_points = [i/350 for i in learning_rate_decay_points]
# parse that to out number of epochs
learning_rate_decay_points = [int(np.ceil(i * epochs))
                              for i in relative_decay_points]


data = cga.MNIST()

pickling_ga.run('', data, folder,
                population_size=population_size,
                search_space=search_space,
                epoch_fn=const_epochs_6,
                learning_rate_decay_points=learning_rate_decay_points)

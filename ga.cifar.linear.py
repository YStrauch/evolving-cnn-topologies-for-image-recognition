import numpy as np

import cga

import pickling_ga
from search_space_reimpl import search_space


folder = 'experiments/'

# Set up evaluation
population_size = 20

maxepochs = 70


def linear_epochs_30_70(individual, gen_index, prev_evaluations):
    start = 30
    end = maxepochs
    maxgens = 20

    # y = ax + b
    diff = end - start
    b = start
    a = diff / (maxgens - 1)

    return round(a * gen_index + b)


# original paper
learning_rate_decay_points = [1, 149, 249]

relative_decay_points = [i/350 for i in learning_rate_decay_points]
# parse that to out number of epochs
learning_rate_decay_points = [int(np.ceil(i * maxepochs))
                              for i in relative_decay_points]


data = cga.CIFAR10()

pickling_ga.run('', data, folder,
                population_size=population_size,
                search_space=search_space,
                epoch_fn=linear_epochs_30_70,
                learning_rate_decay_points=learning_rate_decay_points)

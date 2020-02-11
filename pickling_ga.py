'''Helper function that pickles / unpickles data and runs the GA'''
import os

import logging

import cga


def run(prefix, data, folder, search_space, epoch_fn,
        population_size=20, crossover_probability=.9, mutation_probability=.2, fitness_punishment_per_hour=0,
        learning_rate=0.1, learning_rate_decay_points=None, learning_rate_decay_factor=.9, momentum=0.9,
        gpus=None, number_of_cpus=0,
        create_individual=None
        ):

    learning_rate_decay_points = learning_rate_decay_points or []

    create_individual = create_individual if create_individual else lambda: cga.CNN.random(
        input_shape=data.example.shape,
        num_classes=data.num_classes,
        search_space=search_space
    )

    def restore_individual(topology):
        return cga.CNN.from_string(
            topology,
            input_shape=data.example.shape,
            num_classes=data.num_classes
        )

    folder = '%s/%sdata-%s__popsize-%d__crossover-%s__mutation-%s__punishment-per-hour-%s__learning-%s__epochfn-%s' % (
        folder,
        prefix,
        data.__class__.__name__,
        population_size,
        crossover_probability,
        mutation_probability,
        fitness_punishment_per_hour,
        '_'.join([str(learning_rate), str(learning_rate_decay_factor),
                  str(learning_rate_decay_points), str(momentum)]),
        epoch_fn.__name__
    )
    print('Experiment: %s' % folder)
    pickle = '%s/algorithm.p' % folder

    os.makedirs(folder, exist_ok=True, mode=0o700)

    logging.basicConfig(
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        filename='%s/out.log' % folder,
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    if gpus is not None or number_of_cpus is not None:
        cga.HardwareManager.set_devices(
            number_of_cpus=number_of_cpus, gpus=gpus
        )

    try:
        algorithm = cga.GA.from_pickle(pickle,
                                       data=data,
                                       restore_individual=restore_individual,
                                       epoch_fn=epoch_fn
                                       )
        # algorithm.pickle(pickle)
        logging.info('Restored pickled history')
        best_fitness = algorithm.best_fitness
        logging.info('Best fitness (from pickle): %.3f%%', 100*best_fitness)
        logging.info(algorithm.best_individual.topology)

    except FileNotFoundError:
        logging.info('Did not restore pickled history')
        best_fitness = -1

        evaluator = cga.Evaluator(
            data,
            learning_rate=learning_rate,
            learning_rate_decay_factor=learning_rate_decay_factor,
            learning_rate_decay_points=learning_rate_decay_points,
            momentum=momentum,
            fitness_punishment_per_hour=fitness_punishment_per_hour,
        )
        algorithm = cga.GA(evaluator=evaluator,
                           create_individual=create_individual,
                           population_size=population_size,
                           crossover_probability=crossover_probability,
                           mutation_probability=mutation_probability,
                           epoch_fn=epoch_fn,
                           pickle_file=pickle
                           )

    start_generation = len(algorithm.generations)

    for generation in range(start_generation, 20):
        logging.info('Generation %d', generation)

        algorithm.evolve(search_space, elitism=True)
        if algorithm.best_fitness > best_fitness:
            best_fitness = algorithm.best_fitness
            logging.info('Found better fitness: %.3f%%', 100*best_fitness)
            logging.info('Best topology:\n%s',
                         algorithm.best_individual.topology)

        algorithm.pickle(pickle)

    logging.info('FINISHED')
    # algorithm.pickle(pickle)

    logging.info('Best fitness (ever): %.3f%%', 100*best_fitness)
    logging.info('Best topology (ever):\n%s',
                 algorithm.best_individual.topology)

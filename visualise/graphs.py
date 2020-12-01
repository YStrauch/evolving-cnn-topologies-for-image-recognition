import sys
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re

try:
    import cga
except ModuleNotFoundError:
    sys.path.insert(0, os.path.curdir)
    import cga


data = cga.MNIST()  # not used, so doesn't matter if its CIFAR or MNIST


def load_generations(experiment):
    def restore_individual(topology):
        return cga.CNN.from_string(
            topology,
            input_shape=(32, 32),
            num_classes=10
        )

    algorithm = cga.GA.from_pickle(
        'experiments/%s/algorithm.p' % experiment,
        data=data,
        restore_individual=restore_individual
    )

    generations = algorithm.generations
    for gen in generations:
        gen.recalc()

    return generations


def plot_stats(generations, out, base_gens=None):

    if not base_gens:
        fake_pop = cga.Population(evaluator=cga.Evaluator(data=cga.CIFAR10()))

    max_fitness = np.zeros(len(generations) + 1)
    min_fitness = np.zeros(len(generations) + 1)
    avg_fitness = np.zeros(len(generations) + 1)
    var_fitness = np.zeros(len(generations) + 1)
    base_avg_fitness = np.zeros(len(generations) + 1)

    max_depth = np.zeros(len(generations) + 1)
    min_depth = np.zeros(len(generations) + 1)
    avg_depth = np.zeros(len(generations) + 1)
    var_depth = np.zeros(len(generations) + 1)
    base_avg_depth = np.zeros(len(generations) + 1)

    max_params = np.zeros(len(generations) + 1)
    min_params = np.zeros(len(generations) + 1)
    avg_params = np.zeros(len(generations) + 1)
    base_avg_params = np.zeros(len(generations) + 1)

    pool_ratio = np.zeros(len(generations) + 1)
    base_pool_ratio = np.zeros(len(generations) + 1)

    for gen_index, generation in enumerate(generations):
        gen_index += 1

        base_gen = base_gens[gen_index-1] if base_gens else fake_pop

        # fitness
        fitnesses = [ev.fitness for ev in generation.evaluations]
        base_fitnesses = [ev.fitness for ev in base_gen.evaluations]
        avg_fitness[gen_index] = np.average(fitnesses)
        base_avg_fitness[gen_index] = np.average(base_fitnesses) if base_fitnesses else 0
        var_fitness[gen_index] = np.var(fitnesses)
        max_fitness[gen_index] = max(fitnesses)
        min_fitness[gen_index] = min(fitnesses)

        # depth
        depths = [len(ind.topology.split('\n')) - 1
                  for ind in generation.individuals]
        base_depths = [len(ind.topology.split('\n')) - 1
                       for ind in base_gen.individuals]
        avg_depth[gen_index] = np.average(depths)
        base_avg_depth[gen_index] = np.average(base_depths) if base_depths else 0
        var_depth[gen_index] = np.var(depths)
        max_depth[gen_index] = max(depths)
        min_depth[gen_index] = min(depths)

        # Number of params
        params = [sum(p.numel() for p in ind.parameters())
                  for ind in generation.individuals]
        base_params = [sum(p.numel() for p in ind.parameters())
                       for ind in base_gen.individuals]
        avg_params[gen_index] = np.average(params)
        base_avg_params[gen_index] = np.average(base_params) if base_params else 0
        max_params[gen_index] = max(params)
        min_params[gen_index] = min(params)

        layer_types = [layer[0:1]
                       for ind in generation.individuals
                       for layer in ind.topology.split('\n')[:-2]
                       ]
        base_layer_types = [layer[0:1]
                       for ind in base_gen.individuals
                       for layer in ind.topology.split('\n')[:-2]
                       ]
        num_pools = len([1 for l in layer_types if l == 'P'])
        base_num_pools = len([1 for l in base_layer_types if l == 'P'])

        pool_ratio[gen_index] = num_pools / len(layer_types)
        if len(base_layer_types):
            base_pool_ratio[gen_index] = base_num_pools / len(base_layer_types)

    w, h = plt.figaspect(5/6)
    fig, axs = plt.subplots(2, 2, figsize=(w*1.5, h*1.5))
    axs = axs.reshape(-1)

    x = range(1, len(generations) + 1)

    def plot(i, avg, var, max, min, title):
        plt_avg, = axs[i].plot(x, avg[1:])
        plt_max, = axs[i].plot(x, max[1:])
        plt_min, = axs[i].plot(x, min[1:])

        plt_var = None
        if var is not None and avg is not None:
            plt_var = axs[i].errorbar(x, avg[1:], var[1:],
                                      linestyle='None', capsize=3)

        axs[i].set_title(title)
        axs[i].set_xticks(range(1, len(avg), 2))
        axs[i].set_xlabel('Generation')

        # shrink for legend
        box = axs[i].get_position()
        axs[i].set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])

        lines = [plt_avg, plt_max, plt_min]
        if plt_var:
            lines.append(plt_var)
        return lines

    lines = plot(0, avg_fitness, None,
                 max_fitness, min_fitness, 'Fitness')
    plot(1, avg_depth, None, max_depth, min_depth, 'CNN Depth')
    plot(2, avg_params, None, max_params, min_params, 'Number of Parameters')
    axs[2].set_yscale('log')

    lines.append(axs[3].plot(x, pool_ratio[1:], c='black')[0])
    axs[3].set_title('Proportion of Pooling Layers')
    axs[3].set_xticks(range(1, len(pool_ratio), 2))
    axs[3].set_ylim([0, 1])
    axs[3].axhline(y=.5, dashes=(1, 2), c='grey')
    axs[3].set_xlabel('Generation')

    # plot base
    if base_gens is not None:
        axs[0].plot(x, base_avg_fitness[1:], color='grey', linestyle='dashed')
        axs[1].plot(x, base_avg_depth[1:], color='grey', linestyle='dashed')
        axs[2].plot(x, base_avg_params[1:], color='grey', linestyle='dashed')
        base, = axs[3].plot(x, base_pool_ratio[1:], color='grey', linestyle='dashed')

        lines += [base]

    plt.figlegend(lines, ['Average', 'Max', 'Min', 'Percentage', 'Average Exp1'],
                  loc='center right',
                  #   ncol=4, labelspacing=0.,
                  )

    plt.tight_layout(rect=(0, 0, 5/6, 1), h_pad=3)

    plt.savefig(out)
    # plt.show()



def plot_mutations(generations, out):
    mutation_improvements_x = []
    mutation_improvements_y = []
    crossover_improvements_x = []
    crossover_improvements_y = []
    mutation_and_crossover_improvements_x = []
    mutation_and_crossover_improvements_y = []

    # improvements per mutation type
    improvement_per_mutation = {}

    for gen_index, generation in enumerate(generations[1:]):
        gen_index += 1

        for individual in generation.individuals:
            history = individual.history
            if history.reason == 'clone':
                continue

            history_p1 = history.parent1.history
            history_p2 = history.parent2.history if history.parent2 else None

            parent_fitnesses = []
            if history_p1.evaluation:
                parent_fitnesses.append(history_p1.evaluation.fitness)
            if history_p2 and history_p2.evaluation:
                parent_fitnesses.append(history_p2.evaluation.fitness)

            # cross-over
            if history.reason[:10] == 'cross_over':
                crossover_improvements_x.append(gen_index)
                crossover_improvements_y.append(
                    individual.history.evaluation.fitness -
                    np.mean(parent_fitnesses)
                )
                continue

            # --- mutation ---
            if parent_fitnesses:
                # parent has been evaluated => only mutation, no cross-over before
                improvement = individual.history.evaluation.fitness - \
                    parent_fitnesses[0]
                mutation_improvements_x.append(gen_index)
                mutation_improvements_y.append(improvement)

                reason = history.reason

            else:
                # cross-over, then mutation
                # the parent has no fitness, so use parents from parent1
                grandparent_fitnesses = [
                    grandparent.history.evaluation.fitness for grandparent in [history_p1.parent1, history_p1.parent2]
                ]

                improvement = individual.history.evaluation.fitness - \
                    np.mean(grandparent_fitnesses)

                mutation_and_crossover_improvements_x.append(gen_index)
                mutation_and_crossover_improvements_y.append(improvement)

                reason = 'cross_over + %s' % history.reason

            if reason not in improvement_per_mutation:
                improvement_per_mutation[reason] = {
                    'x': [],
                    'y': []
                }
            improvement_per_mutation[reason]['x'].append(gen_index)
            improvement_per_mutation[reason]['y'].append(improvement)

    w, h = plt.figaspect(3/5)
    fig, axs = plt.subplots(1, 2, figsize=(w, h))
    axs = axs.reshape(-1)

    axs[0].scatter(mutation_improvements_x,
                   mutation_improvements_y,
                   label='Mutation',
                   alpha=.45
                   )
    axs[0].scatter(crossover_improvements_x,
                   crossover_improvements_y,
                   label='Crossover',
                   alpha=.45
                   )
    axs[0].scatter(mutation_and_crossover_improvements_x,
                   mutation_and_crossover_improvements_y,
                   label='Crossover + Mutation',
                   alpha=.45
                   )

    for mutation in improvement_per_mutation:
        x = improvement_per_mutation[mutation]['x']
        y = improvement_per_mutation[mutation]['y']

        # Make the label nicer to read
        replace = {
            'cross_over_1': 'Crossover',
            'cross_over_2': 'Crossover',
            'cross_over': 'Crossover',
            'insert_layer_': 'Insert ',
            'change_layer': 'Mutate Layer',
            'remove_layer': 'Remove Layer',
            'mutate_clf': 'Mutate Classifier',
            'mutate_epoch': 'Mutate Epoch'
        }

        label = mutation
        for key in replace:
            label = label.replace(key, replace[key])

        axs[1].scatter(x, y, label=label, alpha=.45)

    legend_locs = [
        'lower left',
        'lower right',
    ]
    for i, ax in enumerate(axs):
        ax.axhline(y=0, dashes=(1, 2), c='grey')
        ax.set_xlim([0, len(generations)])
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness improvement')

        # order legend alphabetically
        handles, labels = ax.get_legend_handles_labels()
        order = np.argsort(labels)
        plt.figlegend([handles[idx] for idx in order],
                      [labels[idx] for idx in order],
                      loc=legend_locs[i])

    axs[0].set_title('All Genetic Operations')
    axs[1].set_title('Involving Mutation')

    plt.tight_layout(rect=(0, 1/4, 1, 1), h_pad=3)
    plt.savefig(out)
    # plt.show()


def plot_layers(generations, out):

    investigate_generations = [1, 10, 20]

    # for experiments that are WIP: do not plot generations that have not been evaluated yet
    investigate_generations = [
        gen for gen in investigate_generations if gen <= len(generations)
    ]

    w, h = plt.figaspect(1.1 / len(investigate_generations))
    fig, axs = plt.subplots(1, len(investigate_generations),
                            sharey=True, figsize=(w*.6, h*.6))
    axs = axs.reshape(-1)

    unique_layer_types = np.unique([
        layer[0:1] for generation in generations for ind in generation.individuals for layer in ind.topology.split('\n')[:-2]
    ])  # ['P', 'S', 'C']

    # make it nice for the legend
    legend = [('Pool' if l == 'P' else 'Skip' if l ==
               'S' else 'Conv' if l == 'C' else 'Other') + ' Layer' for l in unique_layer_types]

    for fig_index, gen_index in enumerate(investigate_generations):
        gen = generations[gen_index - 1]

        max_depth = max([len(ind.topology.split('\n')) - 2  # -2 because we don't count the classifier
                         for ind in gen])

        # this array holds the count for each layer type per depth
        count_layer_types = np.zeros((max_depth, len(unique_layer_types)))

        for ind in gen:
            layers = [layer[0:1]
                      for layer in ind.topology.split('\n')[:-2]
                      ]  # ['S', 'P', 'P', 'C', ...]
            layers_indizes = np.searchsorted(
                unique_layer_types, layers
            )  # [1, 0, 0, 2, ...]

            # count each layer type
            for layer_index in np.unique(layers_indizes):
                of_layer_type = np.where(
                    layers_indizes == layer_index, 1, 0
                )  # [0, 1, 1, 0, ...] for 'P'

                # if the network is smaller than max length, add zeroes to the end
                of_layer_type = np.append(
                    of_layer_type, np.zeros(max_depth - len(of_layer_type))
                )

                # add it up
                count_layer_types[:, layer_index] += of_layer_type

        lines = axs[fig_index].plot(range(max_depth), count_layer_types,
                                    label=unique_layer_types)
        axs[fig_index].set_xlabel('CNN layer index')
        axs[fig_index].set_ylabel('Number of layers')
        axs[fig_index].set_title('Generation %s' % gen_index)
        # force whole numbers on x axis
        axs[fig_index].xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True)
        )

    plt.figlegend(iter(lines), legend,
                  loc='lower center',
                  ncol=len(unique_layer_types), labelspacing=0.
                  )

    plt.tight_layout(rect=(0, 1/10, 1, 1), h_pad=3)

    # plt.show()
    plt.savefig(out)


def plot_time_diff(exp, comparisons, out):
    w, h = plt.figaspect(3/4)
    fig, ax = plt.subplots(figsize=(w*0.75, h*0.75))

    exp = load_generations(exp)
    gens_comparisons = [load_generations(file_name)
                        for file_name, exp_name in comparisons]

    diffs = np.zeros((len(comparisons), len(exp)))
    for gen_index in range(len(exp)):
        duration_exp = sum(
            [ev.duration for ev in exp[gen_index].evaluations]
        )

        duration_comparison = [sum(
            [ev.duration for ev in comparison[gen_index].evaluations]
        ) for comparison in gens_comparisons]

        for comp_index in range(len(comparisons)):
            diffs[comp_index][gen_index] = duration_comparison[comp_index] - duration_exp

    for i, diff in enumerate(diffs):
        ax.plot(range(1, len(exp) + 1), diff,
                label=comparisons[i][1].replace('exp', 'Experiment '))

    if len(diffs) > 1:
        ax.legend()

    ax.set_xlabel('Generation')
    ax.set_ylabel('Time Improvement (hours)')

    # force whole numbers on x axis
    ax.set_xticks(range(1, 20, 2))

    plt.tight_layout()
    plt.savefig(out)
    # plt.show()


def plot_layer_configs(generations, out):

    # regex to extract filter size
    regex_filter = '([0-9]+)\('
    regex_kernel = 'P (\w+)\['

    # get list of unique filter sizes, i.e. [64, 128, 256]
    topologies = '\n'.join(
        np.unique([ind.topology for gen in generations for ind in gen]))
    unique_conv_filter_sizes = np.array(
        np.unique(re.findall(regex_filter, topologies)), dtype=np.int)
    unique_pool_functions = np.array(
        np.unique(re.findall(regex_kernel, topologies)))

    # initialise the count of the filters per generation to 0
    filters_per_gen = np.zeros(
        (len(unique_conv_filter_sizes), len(generations)))
    functions_per_gen = np.zeros(
        (len(unique_pool_functions), len(generations)))

    # build topologies per to gen
    topologies = [[ind.topology for ind in gen] for gen in generations]

    # extract filter sizes and kernel functions
    filter_sizes = [[np.array(re.findall(regex_filter, topology), dtype=np.int)
                     for topology in gen] for gen in topologies]
    kernel_functions = [[np.array(re.findall(regex_kernel, topology))
                         for topology in gen] for gen in topologies]

    # count the number of filters/kernels per gen so we can normalise
    num_kernels = np.zeros(len(generations))
    num_filters = np.zeros(len(generations))
    for gen_index, gen in enumerate(generations):
        for ind in gen.individuals:
            num_filters[gen_index] += len(
                re.findall(regex_filter, ind.topology))
            num_kernels[gen_index] += len(
                re.findall(regex_kernel, ind.topology))

    # count filters per individual
    for filter_index, filter_size in enumerate(unique_conv_filter_sizes):
        for gen_index, gen in enumerate(generations):
            filter_count = 0
            for ind_filters in filter_sizes[gen_index]:
                filter_count += len(np.nonzero(filter_size == ind_filters)[0])

            filters_per_gen[filter_index][gen_index] = filter_count / \
                num_filters[gen_index]

    # count kernels per individual
    for filter_index, kernel_function in enumerate(unique_pool_functions):
        for gen_index, gen in enumerate(generations):
            kernel_count = 0
            for ind_kernels in kernel_functions[gen_index]:
                kernel_count += len(np.nonzero(kernel_function ==
                                               ind_kernels)[0])

            functions_per_gen[filter_index][gen_index] = kernel_count / \
                num_kernels[gen_index]

    w, h = plt.figaspect(1/2)
    fig, axs = plt.subplots(1, 2, figsize=(w*.8, h*.8))
    axs = axs.reshape(-1)

    for i, curve in enumerate(filters_per_gen):
        axs[0].plot(range(1, len(generations) + 1), curve,
                    label=unique_conv_filter_sizes[i])

    for i, curve in enumerate(functions_per_gen):
        axs[1].plot(range(1, len(generations) + 1),
                    curve, label=unique_pool_functions[i])

    titles = ['Number of Skip Filters', 'Pool Kernel Function']
    for i, ax in enumerate(axs):
        ax.legend()
        ax.set_title(titles[i])
        ax.set_xticks(range(1, len(generations), 2))
        ax.set_xlabel('Generation')

    axs[0].set_ylabel('Proportion per skip layer')
    axs[1].set_ylabel('Proportion per pool layer')

    plt.tight_layout()
    plt.savefig(out)
    plt.show()


def plot(experiment, out, base_generations):
    generations = load_generations(experiment)

    plot_stats(generations, 'visualise/%sstats.pdf' % out, base_generations)
    plot_mutations(generations, 'visualise/%smutations.pdf' % out)
    plot_layers(generations, 'visualise/%slayers.pdf' % out)
    plot_layer_configs(generations, 'visualise/%slayerconfigs.pdf' % out)


# base experiment
exp1 = (
    'data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0__learning-0.1_0.9_[1, 26, 43]_0.9__epochfn-const_epochs_60',
    'exp1'
)

# approach 2
exp2 = (
    'approach-2__data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0__learning-0.1_0.9_[1, 26, 43]_0.9__epochfn-const_epochs_60',
    'exp2'
)

# approach 3
exp3 = (
    'approach-3__data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0__learning-0.1_0.9_[1, 26, 43]_0.9__epochfn-const_epochs_60',
    'exp3'
)

# base mnist
exp4 = (
    'data-MNIST__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0__learning-0.1_0.9_[1, 3, 5]_0.9__epochfn-const_epochs_6',
    'exp4'
)

# regularisation
exp5 = (
    'data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0.05__learning-0.1_0.9_[1, 26, 43]_0.9__epochfn-const_epochs_60',
    'exp5'
)

# linear epochs
exp6 = (
    'data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0__learning-0.1_0.9_[1, 30, 50]_0.9__epochfn-linear_epochs_30_70',
    'exp6'
)

# new experiment, not part of thesis
# regularisation and linear epochs
exp7 = (
    'data-CIFAR10__popsize-20__crossover-0.9__mutation-0.2__punishment-per-hour-0.05__learning-0.1_0.9_[1, 30, 50]_0.9__epochfn-linear_epochs_30_70',
    'exp7'
)

exps = [
    exp1,
    exp2,
    exp4,
    exp3,
    exp5,
    exp6,
    exp7,
]

base_generations = load_generations(exp1[0])

for exp in exps:
    plot(*exp, base_generations if exp != exp1 else None)

for exp in [exp5, exp6, exp7]:
    plot_time_diff(exp[0], [exp1], 'visualise/%stimediff.pdf' % exp[1])

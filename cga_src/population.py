import numpy as np

from cga_src.evaluator import Evaluator


class Population():
    def __init__(self, evaluator, individuals=None):
        assert isinstance(evaluator, Evaluator)
        self._individuals = individuals or []
        self.evaluator = evaluator
        self._evaluations = [None for i in self._individuals]
        self._recalc_fitness = True

    def recalc(self):
        if self._recalc_fitness:
            if self._individuals:
                promises = []
                for individual in self._individuals:
                    promises.append(self.evaluator.get_evaluation(individual))

                # this will block for one promise after another, but the promises are still
                # evaluated asynchronously so it doesn't matter
                self._evaluations = [promise.result() for promise in promises]
            else:
                self._evaluations = []
            self._recalc_fitness = False

    @property
    def best_individual(self):
        self.recalc()
        return self._individuals[np.argmax([e.fitness for e in self._evaluations])]

    @property
    def best_fitness(self):
        self.recalc()
        return max([e.fitness for e in self._evaluations])

    @property
    def worst_individual(self):
        self.recalc()
        return self._individuals[np.argmin([e.fitness for e in self._evaluations])]

    @property
    def worst_fitness(self):
        self.recalc()
        return min([e.fitness for e in self._evaluations])

    @property
    def evaluation_time(self):
        if not self.evaluations:
            return 0
        return sum([e.duration for e in self.evaluations])

    @property
    def individuals(self):
        return self._individuals

    @property
    def evaluations(self):
        # self.recalc() # do not recalc for pickling process in between!
        return self._evaluations

    def tournament_selection(self, select_worse=False):
        assert len(self._individuals) > 1

        self.recalc()
        fitnesses = [e.fitness for e in self._evaluations]
        index1 = np.random.randint(0, len(self._individuals))
        index2 = index1

        while index2 == index1:
            index2 = np.random.randint(0, len(self._individuals))

        better = self._individuals[
            index1 if fitnesses[index1] > fitnesses[index2] else index2
        ]

        worse = self._individuals[
            index1 if fitnesses[index1] < fitnesses[index2] else index2
        ]

        return worse if select_worse else better

    def append(self, individual):
        self._individuals.append(individual)
        self._evaluations.append(None)
        self._recalc_fitness = True

    def as_dict(self):
        self.recalc()
        return {
            individual: evaluation
            for individual, evaluation in zip(self._individuals, self._evaluations)
        }

    # methods to make this class behave like a list, except it will take track of fitness

    def __repr__(self):
        return self._individuals.__repr__()

    def __reversed__(self):
        return self._individuals.__reversed__()

    def __iter__(self):
        return self._individuals.__iter__()

    def __len__(self):
        return self._individuals.__len__()

    def __length_hint__(self):
        return self._individuals.__length_hint__()

    def __contains__(self, item):
        return self._individuals.__contains__(item)

    def __getitem__(self, key):
        return self._individuals.__getitem__(key)

    def __delitem__(self, key):
        self._individuals.__delitem__(key)
        self._evaluations.__delitem__(key)

    def __setitem__(self, key, value):
        self._individuals.__setitem__(key, value)
        self._evaluations.__setitem__(key, None)
        self._recalc_fitness = True

    def index(self, value):
        return self._individuals.index(value)

    def __add__(self, other):
        if isinstance(other, Population):
            other = other._individuals  # pylint: disable=protected-access

        assert isinstance(other, list)

        return Population(
            self.evaluator,
            individuals=self._individuals + other,
        )

try:
    from atorch.auto.engine.sg_algo.bayes_opt_sg import BOAlgorithm

    bo_algo_is_available = True
except ModuleNotFoundError:
    bo_algo_is_available = False
from atorch.auto.engine.sg_algo.combination_sg import CombinationAlgorithm


class StrategyGenerationAlgorithmLibrary(object):
    """
    Each strategy generation (SG) algorithm is a StrategyGenerationAlgorithm
    instance, which can be called to generate new strategies.
    """

    def __init__(self):
        self.algorithms = {}
        self.add_algorithms()

    def add_algorithms(self):
        algo = CombinationAlgorithm()
        self.algorithms[algo.name] = algo
        if bo_algo_is_available:
            bo_algo = BOAlgorithm()
            self.algorithms[bo_algo.name] = bo_algo

    def __getitem__(self, name):
        if name in self.algorithms:
            return self.algorithms[name]
        return None

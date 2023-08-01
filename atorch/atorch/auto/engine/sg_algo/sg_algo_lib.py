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

    def __getitem__(self, name):
        if name in self.algorithms:
            return self.algorithms[name]
        return None

class StrategyGenerationAlgorithm(object):
    """A strategy generation (SG) algorithm implementation.
    Call strategy_generate with executor to generate candidate strategies.
    strategy_generate can be called multiple times to generate strategies in
    multiple stages.
    """

    def __init__(self, name=None):
        self.name = name
        self.is_done = False

    def strategy_generate(self, _):
        """
        Input: executor which contains optimization method, strategies.
        The output is 3-tuple:
        is_done: bool incidating if the algorithm finishs after this call.
        tasks: None or list(task), new tasks to execute.
        new_strategy_num: int for the number of new strategy added.
        """
        self.is_done = True
        return self.is_done, None, 0

    def __call__(self, executor):
        return self.strategy_generate(executor)

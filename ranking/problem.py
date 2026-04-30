import pandas as pd

class RankingProblem:
    """
    A class to represent a top-k ranking problem.

    Attributes
    ----------
    candidates : dataframe of candidates
        All candidates of the ranking data with relevance scores and groups..
    k : int
        The number of candidates to select.
    proportional_target : dict
        The proportional target of each group in the data.
    exposure_target : dict
        The exposure target of each group in the data.
    """
    def __init__(self, candidates, group_col, relevance_col, k, proportional_target):
        self.candidates = candidates
        self.group_col = group_col
        self.relevance_col = relevance_col
        self.k = k
        self.rank = [None] * len(self.candidates)
        self.group_mapping = {}
        self.proportional_target = proportional_target
        self.groups = self.candidates[group_col].unique()

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        if not isinstance(value, int):
            raise ValueError("Number expected")
        elif not value > 0 and value <= len(self.candidates):
            raise ValueError("Positiv number smaller than number of candidates expected")
        self._k = value

    @property
    def proportional_target(self):
        return self._proportional_target

    @proportional_target.setter
    def proportional_target(self, value):
        if not isinstance(value, dict):
            raise ValueError("Dict expected")
        elif not all(0 <= v <= 1 for v in value.values()):
            raise ValueError("Proportion between 0 and 1 for each group expected")
        elif not sum(value.values()) <= 1:
            raise ValueError("Proportion sum smaller than 1 expected")
        self._proportional_target = {k: v for k, v in value.items()}

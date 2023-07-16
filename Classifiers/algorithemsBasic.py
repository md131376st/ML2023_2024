from Data.Info import Info


class AlgorithmBasic:
    def __init__(self, info=None, prior=1/2):
        # if selected for final modeling
        if info is None:
            self.info = Info()
            self.prior = prior
        else:
            self.info = info
            self.prior = prior

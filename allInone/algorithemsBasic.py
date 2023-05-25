from Info import Info


class AlgorithmBasic:
    def __init__(self, info=None):
        # if selected for final modeling
        if info is None:
            self.info = Info()
        else:
            self.info = info

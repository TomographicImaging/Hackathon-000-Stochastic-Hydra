from cil.optimisation.functions import IndicatorBox

class indicator_func(object):

    def __init__(self, name):
        self.name = name
        self.func = IndicatorBox(lower=0)

    def get_func(self):
        return IndicatorBox(lower=0)


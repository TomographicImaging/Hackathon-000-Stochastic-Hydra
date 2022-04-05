from cil.optimisation.functions import KullbackLeibler

class gd(object):

    def __init__(self, name):
        self.name = name

    def get_func(self, data):
        return KullbackLeibler(b=data)



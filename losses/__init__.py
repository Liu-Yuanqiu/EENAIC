import imp
from losses.cross_entropy import CrossEntropy

__factory = {
    'CrossEntropy': CrossEntropy,
}

def names():
    return sorted(__factory.keys())

def create(name):
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name]()
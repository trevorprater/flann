from pyflann import *
from numpy import *
from numpy.random import *

dataset = rand(1000, 128)
testset = rand(10, 128)

flann = FLANN()
params = flann.build_index(dataset, algorithm="autotuned", target_precision=0.9, log_level="info")

result, dists = flann.nn_index(testset, 5, checks=params['checks'])

flann.add_points(testset, 2)
_result, _ = flann.nn_index(testset, 5, checks=params['checks'])


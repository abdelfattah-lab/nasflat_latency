help_n = [0.948,0.410,0.604,0.509,0.729,0.746]
multipredict_n = [0.930,0.820,0.907,0.757,0.947,0.952]
flatnas_n = [0.959,0.893,0.967,0.857,0.962,0.959]

help_f = [0.91,0.37,0.793,0.543,0.413,0.799]

multipredict_f = [0.960,0.45,0.756,0.567,0.434,0.763]

flatnas_f = [0.961,0.577,0.809,0.871,0.814,0.734]

import numpy as np

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

print(geo_mean(help_n))
print(geo_mean(multipredict_n))
print(geo_mean(flatnas_n))

print(geo_mean(help_f))
print(geo_mean(multipredict_f))
print(geo_mean(flatnas_f))

import numpy as np
from numpy import inf

def normalize(v, method='norm'):
    if method == 'norm':
        for j in range(np.shape(v)[1]):
            vec = [x[j] for x in v]
            maxi = max(vec)
            mini = min(vec)
            for i in range(len(v)):
                if maxi != mini:
                    v[i][j] = (vec[i] - mini)/(maxi - mini)
                else:
                    v[i][j] = 1
    return v
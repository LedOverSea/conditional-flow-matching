import ot as pot
import numpy as np
if __name__ == "__main__":
    #a, b, M, reg
    
    a = [.1, .2]
    b = [.1, .1]
    M = [[0., 1.], [2., 3.]]
    
    print(pot.partial.entropic_partial_wasserstein(a, b, M, 1, 0.1), 2)
    print(np.round(pot.partial.entropic_partial_wasserstein(a, b, M, 1, 0.1), 2))
    # array([[0.06, 0.02],
    #        [0.01, 0.  ]])
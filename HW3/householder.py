import numpy as np

def householder(x):
    v = x.reshape(x.size, 1)

    sigma = np.sign(v[0, 0]) * np.linalg.norm(v)
    u = v + sigma * np.eye(v.size, 1)
    u = u / np.linalg.norm(u)
    
    Px = np.identity(u.size) - 2 * u @ u.T
    
    return Px
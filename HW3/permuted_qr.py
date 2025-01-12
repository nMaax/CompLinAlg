import numpy as np

def permuted_qr(A):

    m, n = A.shape

    if m != n:
        raise ValueError(f"Matrix must be square to perform QR factorization, shape (m={m}, n={n}) was given.")

    Q = np.eye(n)
    R = np.eye(n)
    P = np.eye(1, n)

    # TODO...

    return Q, R, P

def givens_mat(X, h, k):
    """
    Function that compute the Givens matrix for a given square matrix X and with respect to row h and column k
    :param X: square matrix represented as 2D-array object (numpy ndarray);
    :param h: integer value in the range of the number of X's rows;
    :param k: integer value in the range of the number of X's columns;
    :return G: the Givens matrix as 2D-array object (numpy ndarray).
    """
    
    m, n = X.shape
    
    if m != n:
        print('MATRIX IS NOT SQUARE!')
        return None

    x = X[k, k]
    y = X[h, k]
    
    # d (denominator of both c and s) can be written as:
    # d = np.sqrt(X[k, k]**2 + X[h, k]**2)
    # But is better (due to numerical problems) to use the hypot function.
    d = np.hypot(x, y)
    
    c = x / d
    s = y / d

    G = np.eye(n, n)
    G[k, k] = c
    G[h, h] = c
    G[h, k] = -s
    G[k, h] = s

    return G

if __name__ == "__main__":
    
    # Example matrix
    C = np.array([  [12, -51, 4],
                    [6, 167, -68],
                    [-4, 24, -41]])

    # Perform permuted QR factorization
    Q, R, P = permuted_qr(C)

    # Print results
    print("Q:\n", Q)
    print("R:\n", R)
    print("P:\n", P)

    # Verify that C[:, P] = Q @ R
    print("Is C @ P = Q @ R? ", np.allclose(C[:, P], Q @ R, atol=1e-8))
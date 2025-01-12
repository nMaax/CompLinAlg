import numpy as np

def permuted_qr(C):

    _, n = C.shape
    Q = np.eye(n)
    R = np.eye(n)
    P = np.eye(1, n)

    # TODO...

    return Q, R, P

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
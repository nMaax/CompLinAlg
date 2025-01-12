import numpy as np
from householder import householder
    
def bidiagonalize(A):

    m, n = A.shape
    if (m < n):
        raise ValueError("Matrix must respect shape s.t m >= n, shape (m={m}, n={n}) was given.")

    B = A.copy()
    P = np.eye(m)
    H = np.eye(n)

    for k in range(n):

        a = B[k:m, k]
        P_tilde = np.eye(m)
        P_tilde[k:m, k:m] = householder(a)
        B = P_tilde @ B
        P = P_tilde @ P
        
        if k <= n-3: # usig n-2 one singular value becomes positive
            b = B[k, k+1:n]
            H_tilde = np.eye(n)
            H_tilde[k+1:n, k+1:n] = householder(b)
            B = B @ H_tilde
            H = H @ H_tilde

    return B, P, H

def gk_bidiagonalize():

    m, n = A.shape
    if (m < n):
        raise ValueError("Matrix must respect shape s.t m >= n, shape (m={m}, n={n}) was given.")

    B = A.copy()
    P = np.eye(m)
    H = np.eye(n)

    for k in range(n):

        a = B[k:m, k]
        P_tilde = np.eye(m)
        P_tilde[k:m, k:m] = householder(a)
        B = P_tilde @ B
        P = P_tilde @ P
        
        if k <= n-3: # usig n-2 one singular value becomes positive
            b = B[k, k+1:n]
            H_tilde = np.eye(n)
            H_tilde[k+1:n, k+1:n] = householder(b)
            B = B @ H_tilde
            H = H @ H_tilde

    return B, P, H

if __name__ == "__main__":
    
    # Test the bidiagonalize function with a sample matrix
    A = np.array([[4, 1, 3], [2, 6, 5], [1, 2, 3], [5, 4, 2]])
    print("\nOriginal matrix A:")
    print(A)

    B, P, H = bidiagonalize(A)

    print("\nBidiagonalized matrix B:")
    print(np.round(B, decimals=8))

    print("\nP @ A @ H == B? ", np.allclose(P @ A @ H, B, atol=1e-8))
    print()
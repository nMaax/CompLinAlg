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

        #k = i

        a = B[k:, k]
        P_transformation = np.eye(m)
        P_transformation[k:, k:] = householder(a)
        B = P_transformation @ B
        P = P_transformation @ P
        
        if k <= n-2:
            b = B[k, k+1:]
            H_transformation = np.eye(n)
            H_transformation[k+1:, k+1:] = householder(b)
            B = B @ H_transformation
            H = H @ H_transformation

    return B, P, H

if __name__ == "__main__":
    
    # Test the bidiagonalize function with a sample matrix
    A = np.array([[4, 1, 3], [2, 6, 5], [1, 2, 3], [5, 4, 2]])
    print("Original matrix A:")
    print(A)

    B, P, H = bidiagonalize(A)

    print("\nBidiagonalized matrix B:")
    print(np.round(B, decimals=8))

    # Verify the result
    print("\nVerification (P @ A @ H):")
    print(np.round(P @ A @ H, decimals=8))

    print("\nP @ A @ H == B? ", np.allclose(P @ A @ H, B, atol=1e-8))
import numpy as np
import scipy as sp
from eigenvalues import eigs
from bidiagonalize import bidiagonalize

class SVD:
    def __init__(self, n_components=None, atol=1e-8):
        self.n_components = n_components
        self.atol = atol

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        U, Sigma, V = self.__compute_svd(X)
        
        if self.n_components:
            U = U[:, :self.n_components]
            Sigma = Sigma[:self.n_components, :self.n_components]        
            V = V[:, :self.n_components]

        if self.atol:
            U[np.abs(U) < self.atol] = 0
            V[np.abs(V) < self.atol] = 0
            Sigma[np.abs(Sigma) < self.atol] = 0

        return U, Sigma, V

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def __compute_svd(self, A):

        m, n = A.shape
        if (m < n):
            raise ValueError("Matrix must respect shape s.t m >= n, shape (m={m}, n={n}) was given.")

        B, _, H = bidiagonalize(A)

        #eigvals, Q_tilde = sp.linalg.eig(B.T @ B)
        evals, Q_tilde = eigs(B.T @ B)  # Symmetric eigen-decomposition

        Q = H @ Q_tilde

        C = A @ Q

        U, R, P = sp.linalg.qr(a=C, pivoting=True)
        Sigma = R
        V = Q[:, P]
        
        return U, Sigma, V

# Test the SVD class
if __name__ == "__main__":
    
    from sklearn.decomposition import TruncatedSVD

    np.random.seed(42)

    # Create a random matrix A
    A = np.random.rand(6, 4)  # 6x4 matrix

    # Print the results
    print("\nMatrix A\n")
    print(A)
    
    # Specify the number of components for the truncated SVD
    n_components = 4

    # *** My SVD *** #
    U, Sigma, V = SVD(None).fit_transform(A)

    print("\n*** My SVD *** ")
    print("\nMatrix U")
    print(U)

    U_orthogonal_check = np.allclose(U.T @ U, np.eye(U.shape[1]))
    print("Is U an orthogonal matrix? ", U_orthogonal_check)

    print("\nMatrix Sigma")
    print(Sigma)
    print("\nMatrix V.T")
    print(V.T)

    V_orthogonal_check = np.allclose(V.T @ V, np.eye(V.shape[1]))
    print("Is V an orthogonal matrix? ", V_orthogonal_check)

    print("\n---\n")
    reconstruction_check = np.allclose(U @ Sigma @ V.T, A)
    print("Is U @ Sigma @ V.T == A? ", reconstruction_check)


    # *** Builtin SVD *** #

    svd = TruncatedSVD(n_components=n_components)
    X_svd = svd.fit_transform(A)

    print("\n*** Builtin SVD *** ")
    print("\nMatrix U")
    print(X_svd)
    print("\nMatrix Sigma")
    print(svd.singular_values_)
    print("\nMatrix V.T")
    print(svd.components_)

    print()
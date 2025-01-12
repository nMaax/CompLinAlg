import numpy as np
import scipy as sp
from Bidiagonalizer import Bidiagonalizer
from permuted_qr import permuted_qr
from sklearn.decomposition import TruncatedSVD

class SVD:
    def __init__(self, A, n_components):
        
        m, n = A.shape
        if (m < n):
            raise ValueError("Matrix must respect shape s.t m >= n, shape (m={m}, n={n}) was given.")
        
        self.A = A
        self.n_components = n_components
        
        self.tol = 1e-8

        self.U, self.sigma, self.V = self.compute_svd()

    def get_decomposition(self):
        return self.U, self.sigma, self.V

    def compute_svd(self):

        # Bidiagonalize the matrix # TODO: turn into function
        B, _, H = Bidiagonalizer(self.A).bidiagonalize()

        # Compute Q 
        _, Q_tilde = np.linalg.eig(B.T @ B)
        Q = H @ Q_tilde

        # Compute C
        C = self.A @ Q

        # Compute the permuted QR factorization of C
        #U, R, P = sp.linalg.qr(C, pivoting=True) #self.permuted_qr(C)
        U, R, P = permuted_qr(C)

        # Select only the first n_components
        U = U[:, :self.n_components]
        U[np.abs(U) < self.tol] = 0

        sigma = R[:self.n_components, :self.n_components]
        sigma[np.abs(sigma) < self.tol] = 0
        
        V = Q[:, P]
        V = V[:, :self.n_components]
        V[np.abs(V) < self.tol] = 0
        
        return U, sigma, V

# Test the SVD class
if __name__ == "__main__":
    
    np.random.seed(42)

    # Create a random matrix A
    A = np.random.rand(6, 4)  # 6x4 matrix

    # Print the results
    print("\nMatrix A\n")
    print(A)
    
    # Specify the number of components for the truncated SVD
    n_components = 6

    # *** My SVD *** #

    # Compute the SVD
    svd = SVD(A, n_components)
    
    # Get the decomposition
    U, sigma, V = svd.get_decomposition()

    print("\n*** My SVD *** ")
    print("\nMatrix U")
    print(U)

    # Check if U is an orthogonal matrix
    U_orthogonal_check = np.allclose(U.T @ U, np.eye(U.shape[1]), atol=svd.tol)
    print("Is U an orthogonal matrix? ", U_orthogonal_check)

    print("\nMatrix Sigma")
    print(sigma)
    print("\nMatrix V.T")
    print(V.T)

    # Check if V is an orthogonal matrix
    V_orthogonal_check = np.allclose(V.T @ V, np.eye(V.shape[1]), atol=svd.tol)
    print("Is V an orthogonal matrix? ", V_orthogonal_check)

    print("\n---\n")
    reconstruction_check = np.allclose(U @ sigma @ V.T, A, atol=svd.tol)
    print("Is U @ sigma @ V.T == A? ", reconstruction_check)


    # *** Builtin SVD *** #

    # svd = TruncatedSVD(n_components=n_components)
    # X_svd = svd.fit_transform(A)

    # print("\n*** Builtin SVD *** ")
    # print("\nMatrix U")
    # print(X_svd)
    # print("\nMatrix Sigma")
    # print(svd.singular_values_)
    # print("\nMatrix V.T")
    # print(svd.components_.T)

    print()
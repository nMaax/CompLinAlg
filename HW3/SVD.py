import numpy as np
import scipy as sp
from Bidiagonal import bidiagonalization
from sklearn.decomposition import TruncatedSVD

class SVD:
    def __init__(self, A, n_components):
        self.A = A
        self.n_components = n_components
        self.V, self.sigma, self.U = self.compute_svd()
    
    def get_decomposition(self):
        return self.V.T, self.sigma, self.U

    def compute_svd(self):

        # Bidiagonalize the matrix # TODO: turn into function
        Bidiagonal = bidiagonalization(self.A)
        B, H = Bidiagonal.bidiagonalize()
        
        # Compute Q
        _, Q_tilde = np.linalg.eig(B.T @ B)
        Q = H @ Q_tilde

        # Compute C
        C = self.A @ Q

        # Compute the permuted QR factorization of C
        U, R, P = sp.linalg.qr(C, pivoting=True) #self.permuted_qr(C)

        # Select only the first n_components
        V = Q[:, P]
        V = V[:, :self.n_components]
        sigma = R[:self.n_components, :self.n_components]
        U = U[:, :n_components]

        return V, sigma, U
    
    # def permuted_qr(self, C):  # TODO: move to another file (givens and householder)
    #     """Compute the permuted QR decomposition of C such that CP = UR,
    #     where R has diagonal entries sorted in decreasing order."""
    #     # Perform QR decomposition
    #     Q, R = np.linalg.qr(C, mode='reduced')

    #     # Compute the permutation to sort the diagonal of R in decreasing order
    #     diag_abs = np.abs(np.diag(R))
    #     sorted_indices = np.argsort(-diag_abs)  # Sort diagonals in descending order

    #     # Apply the permutation to R
    #     P = np.eye(C.shape[1])[:, sorted_indices]  # Create the permutation matrix
    #     R = R @ P  # Permute the columns of R
    #     Q = Q  # Q remains unchanged since it is orthogonal

    #    return Q, R, P

# Test the SVD class
if __name__ == "__main__":
    # Create a random matrix A
    np.random.seed(42)
    A = np.random.rand(6, 4)  # 6x4 matrix

    # Print the results
    print()
    print("Matrix A\n")
    print(A)
    
    # Specify the number of components
    n_components = 3

    # Compute the SVD
    svd = SVD(A, n_components)

    # Get the decomposition
    V, sigma, U = svd.get_decomposition()

    print("\n*** My SVD *** ")
    print("\nMatrix U (reduced to n_components):")
    print(U)
    print("\nMatrix Sigma (reduced to n_components):")
    print(sigma)
    print("\nMatrix V (reduced to n_components):")
    print(V)

    svd = TruncatedSVD(n_components=n_components)
    X_svd = svd.fit_transform(A)

    print("\n*** Builtin SVD *** ")
    print("\nMatrix U (reduced to n_components):")
    print(X_svd)
    print("\nMatrix Sigma (reduced to n_components):")
    print(svd.singular_values_)
    print("\nMatrix V (reduced to n_components):")
    print(svd.components_)

    print()
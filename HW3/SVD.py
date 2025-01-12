import numpy as np
import scipy as sp
from Bidiagonal import bidiagonalization

class SVD:
    def __init__(self, A,n_components):
        self.A = A
        self.n_components = n_components
        self.V,self.sigma = self.compute_svd()

    def compute_svd(self):

        Bidiagonal = bidiagonalization(self.A)
        B, H = Bidiagonal.bidiagonalize()  # Local variables instead of self
        sq_B = B.T @ B
        D, Q_tilde = np.linalg.eig(sq_B)
        Q = H @ Q_tilde
        C = self.A @ Q
        print(f"This is Q:\n {Q}")
        print(f"This is A:\n {self.A}")
    
        # FINO A QUI OK DALL'ISTRUZIONE SUCCESSIVA NON VA, SICURAMENTE C'È QUALCHE ERRORE NELLA FUNZIONE PERMUTED_QR
        # DEVE RITORNARE ANCHE U PERCHÈ IL RISULTATO FINALE DEVE ESSERE MAT = U * SIGMA * V^T
        U, R, P = sp.linalg.qr(C, pivoting=True) #self.permuted_qr(C)

        # Select only the first n_components
        V = Q @ P
        V= V[:, :self.n_components]
        sigma = R
        sigma = sigma[:self.n_components, :self.n_components]

        return V, sigma, U

    def get_decomposition(self):
        return self.V, self.sigma
    
    def permuted_qr(self, C):
        """Compute the permuted QR decomposition of C such that CP = UR,
        where R has diagonal entries sorted in decreasing order."""
        # Perform QR decomposition
        Q, R = np.linalg.qr(C, mode='reduced')

        # Compute the permutation to sort the diagonal of R in decreasing order
        diag_abs = np.abs(np.diag(R))
        sorted_indices = np.argsort(-diag_abs)  # Sort diagonals in descending order

        # Apply the permutation to R
        P = np.eye(C.shape[1])[:, sorted_indices]  # Create the permutation matrix
        R = R @ P  # Permute the columns of R
        Q = Q  # Q remains unchanged since it is orthogonal

        return Q, R, P

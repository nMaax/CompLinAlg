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
        print(Sigma)
        U, Sigma, V = self.fix_signs(U, Sigma, V)
        U, Sigma, V = self.sort_singular_values(U, Sigma, V)
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
        evals, V_tilde = eigs(B.T @ B)  # Symmetric eigen-decomposition

        Q = H @ V_tilde
        C = A @ Q

        U, R, P = sp.linalg.qr(a=C, pivoting=True)
        Sigma = R
        V = Q[:, P]
        
        return U, Sigma, V
    
    def fix_signs(self,U, Sigma, V):
        # Ammettiamo che Sigma sia di shape (r, r), con r = min(m, n)
        r = min(Sigma.shape[0],Sigma.shape[1])
        for i in range(r):
            if Sigma[i, i] < 0:
                Sigma[i, i] = -Sigma[i, i]  # Rendo positivo
                #U[:, i]    = -U[:, i]      # Cambio segno alla colonna i di U
                V[:, i] = -V[:, i]  #(invece di U[:, i], ma non entrambe!)
        
        return U, Sigma, V
    
    def sort_singular_values(self, U, Sigma, V):
        # prendi la diagonale
        diag_S = np.diag(Sigma)
        idx = np.argsort(diag_S)[::-1]         # Ordine decrescente
        Sigma_sorted = np.diag(diag_S[idx])    # Ricostruisci diag con l'ordine
        U_sorted = U[:, idx]
        V_sorted = V[:, idx]
        return U_sorted, Sigma_sorted, V_sorted
    
    

"""
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
    """
if __name__ == "__main__":
    np.random.seed(42)

    # Crea una matrice di test, dimensioni 6x4 per esempio
    A = np.random.rand(6, 4)

    print("\nMatrice A originale:\n", A)

    # Istanzia e applica la tua SVD
    my_svd = SVD(n_components=None)
    U, Sigma, V = my_svd.fit_transform(A)

    print("\nU:\n", U)
    print("\nSigma:\n", Sigma)
    print("\nV:\n", V)
    print("\nV.T:\n", V.T)

    # Ricostruisci la matrice: U @ Sigma @ V.T
    # (Se Sigma è m x n, questa moltiplicazione funziona direttamente.
    #  Se hai Sigma quadrata min(m,n), adattati di conseguenza.)
    A_reconstructed = U @ Sigma @ V.T

    print("\nMatrice ricostruita:\n", A_reconstructed)

    # Verifica la differenza
    diff = np.linalg.norm(A - A_reconstructed)
    print("\nNorma della differenza (A - U Sigma V^T):", diff)

    # Check con np.allclose
    is_close = np.allclose(A, A_reconstructed, atol=1e-8)
    print("Ricostruzione corretta entro tolleranza? ", is_close)
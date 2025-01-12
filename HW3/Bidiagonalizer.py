import numpy as np

class Bidiagonalizer():
    def __init__(self, A):
        
        m, n = A.shape
        if (m < n):
            raise ValueError("Matrix must respect shape s.t m >= n, shape (m={m}, n={n}) was given.")
        
        self.m = m
        self.n = n

        self.B = A
        self.P = np.eye(self.m)
        self.H = np.eye(self.n)

        self.tol = 1e-8
    
    def bidiagonalize(self):
        
        for k in range(self.n):
            P_transformation = np.eye(self.m)
            a = self.B[k:self.m, k]
            Px = self.householder_mat( a)
            P_transformation[k:, k:] = Px
            self.B = P_transformation @ self.B
            self.P = P_transformation @ self.P
            
            if k < self.n-2:
                H_transformation = np.eye(self.n)
                b = self.B[k, k+1:self.n]
                Bx = self.householder_mat_row(b)
                H_transformation[k+1:self.n, k+1:self.n] = Bx
                self.B = self.B @ H_transformation
                self.H = H_transformation @ self.H

        self.B[np.abs(self.B) < self.tol] = 0  
        return self.B, self.P, self.H  
    
    def householder_mat(self,x):
        v = x.reshape(x.size, 1)
        sigma = np.sign(v[0, 0]) * np.linalg.norm(v)
        u = v + sigma * np.eye(v.size, 1)
        u = u / np.linalg.norm(u) 
        Px = np.identity(u.size) - 2 * u @ u.T
        return Px
    
    def householder_mat_row(self, x):
        v = x.reshape(1, x.size)
        sigma = np.sign(v[0, 0]) * np.linalg.norm(v)
        u = v + sigma * np.eye(v.size, 1).T
        u = u / np.linalg.norm(u) 
        Px = np.identity(u.size) - 2 * u.T @ u
        return Px

if __name__ == "__main__":
    
    # Test the bidiagonalize function with a sample matrix
    A = np.array([[4, 1, 3], [2, 6, 5], [1, 2, 3], [5, 4, 2]])
    print("Original matrix A:")
    print(A)

    B, P, H = Bidiagonalizer(A).bidiagonalize()

    print("\nBidiagonalized matrix B:")
    print(B)

    # Verify the result
    print("\nVerification (P @ A @ H.T):")
    print(np.round(P @ A @ H.T, decimals=8))
import numpy as np

class bidiagonalization():
    def __init__(self, A):
        self.B = A
        self.m, self.n = A.shape
        self.P = np.eye(self.m)
        self.H = np.eye(self.n)
        self.tol = 1e-8
        self.bidiagonalize()
        
        
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
        return self.B, self.H  
    def getP(self):
        return self.P
    
    def getH(self):
        return self.H
    
    def getB(self):
        return self.B
    
    def householder_mat(self,x):
        """
        Function that compute the reflection matrix of the Householder method for a given vector x.
        :param x: vector that can be given both as 1D-array object and 2D-array column/row object (numpy ndarray);
        :return Px: Householder reflection matrix as 2D-array object (numpy ndarray).
        """
        v = x.reshape(x.size, 1)
        sigma = np.sign(v[0, 0]) * np.linalg.norm(v)
        u = v + sigma * np.eye(v.size, 1)
        u = u / np.linalg.norm(u) 
        Px = np.identity(u.size) - 2 * u @ u.T
        return Px
    
    def householder_mat_row(self, x):
        """
        Function that compute the reflection matrix of the Householder method for a given vector x.
        :param x: vector that can be given both as 1D-array object and 2D-array column/row object (numpy ndarray);
        :return Px: Householder reflection matrix as 2D-array object (numpy ndarray).
        """
        v = x.reshape(1, x.size)
        sigma = np.sign(v[0, 0]) * np.linalg.norm(v)
        u = v + sigma * np.eye(v.size, 1).T
        u = u / np.linalg.norm(u) 
        Px = np.identity(u.size) - 2 * u.T @ u
        return Px






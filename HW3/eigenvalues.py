import numpy as np

def eigs(A, order="descending"):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Determine the sorting order
    if order == "ascending":
        sorted_indices = np.argsort(eigenvalues)  # Sort in ascending order
    elif order == "descending":
        sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
    else:
        raise ValueError("Invalid order. Use 'ascending' or 'descending'.")
    
    # Sort eigenvalues and eigenvectors
    eigenvalues_sorted = np.diag(eigenvalues[sorted_indices])
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    print(np.allclose(eigenvectors_sorted*eigenvalues_sorted*eigenvectors_sorted, A))

    return eigenvalues_sorted, eigenvectors_sorted

# Example usage
if __name__ == "__main__":
    A = np.array([[4, 2],
                  [1, 3]])
    
    eigenvalues_desc, eigenvectors_desc = eigs(A, order="descending")
    eigenvalues_asc, eigenvectors_asc = eigs(A, order="ascending")
    
    print("Eigenvalues (descending):")
    print(eigenvalues_desc)
    
    print("\nEigenvectors (Q matrix, descending):")
    print(eigenvectors_desc)
    
    print("\nEigenvalues (ascending):")
    print(eigenvalues_asc)
    
    print("\nEigenvectors (Q matrix, ascending):")
    print(eigenvectors_asc)

    print("---")
    
    A = np.random.random((3,3))
    eigenValues, eigenVectors = np.linalg.eig(A)


    # idx = eigenValues.argsort()[::-1]   
    # eigenValues = eigenValues[idx]
    # eigenVectors = eigenVectors[:,idx]

    print(np.allclose(eigenVectors @ eigenValues @ np.linalg.inv(eigenVectors), A))

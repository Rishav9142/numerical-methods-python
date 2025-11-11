import numpy as np

def cholesky_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            sum_val = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                # Diagonal elements
                L[i][j] = (A[i][i] - sum_val) ** 0.5
            else:
                # Off-diagonal elements (fixed line)
                L[i][j] = (A[i][j] - sum_val) / L[j][j]

    return L


# Example matrix (symmetric positive definite)
A = np.array([[4, 12, -16],
              [12, 37, -43],
              [-16, -43, 98]], dtype=float)

L = cholesky_decomposition(A)

print("Input Matrix A:\n", A)
print("\nLower Triangular Matrix L:\n", L)
print("\nCheck A ≈ L × Lᵀ:\n", np.dot(L, L.T))

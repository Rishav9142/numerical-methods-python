import numpy as np

def gram_schmidt(A):

    A = A.astype(float)
    n = A.shape[1]
    Q = np.zeros_like(A)
    R = np.zeros((n, n))

    for i in range(n):

        V = A[:, i]

        for j in range(i):
            R[j, i] = np.dot(Q[:, j], A[:, i])
            V = V - R[j, i] * Q[:, j]

        R[i, i] = np.linalg.norm(V)
        Q[:, i] = V / R[i, i]

    return Q, R

A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

Q, R = gram_schmidt(A)

print("Input Matrix A:\n", A)
print("\nOrthogonal Matrix Q:\n", np.round(Q, 3))
print("\nUpper Triangular Matrix R:\n", np.round(R,3))

print("\nCheck A = Q x R:\n", np.round(np.dot(Q, R), 3))
import numpy as np

def gaussian_elimination(A):

    A = A.astype(float)
    n, m = A.shape

    for i in range(n):
        if A[i][i] == 0:
            for k in range(i + 1, n):
                if A[k][i] != 0:
                    A[[i, k]] = A[[k, i]]
                    break


        pivot = A[i][i]
        if pivot != 0:
            A[i] = A[i] / pivot

        for j in range(i + 1, n):
            factor = A[j][i]
            A[j] = A[j] - factor * A[i]

    return A

A = np.array([[2, 1, -1, 8],
              [-3, -1, 2, -11],
              [-2, 1, 2, -3]], dtype = float)

REF = gaussian_elimination(A)

print("Row Echelon Form (REF):")
print(np.round(REF, 3))
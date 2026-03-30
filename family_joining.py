import numpy as np
from tqdm import tqdm

def GetTreeTopology(D, eps=.1):
    V_a, E = [i for i in range(len(D))], []

    A = (len(D)-2) * D - np.sum(D, axis=0)[:, None] - np.sum(D, axis=0)[None]
    np.fill_diagonal(A, np.inf)

    with tqdm(total=len(V_a)-1, unit="step") as pbar:
        while len(V_a) > 3:
            # index_i, index_j = np.unravel_index(np.argmin(
            #     np.array([[(len(D) - 2) * D[i, j] - np.sum(D[i]) - np.sum(D[j]) for j in V_a if j != i] for i in V_a])),
            #                                     (len(V_a), len(V_a) - 1))
            # i, j = V_a[index_i], [j for j in V_a if j != V_a[index_i]][index_j]

            index_i, index_j = np.unravel_index(np.argmin(A[np.ix_(V_a, V_a)]), (len(V_a),)*2)
            i, j = V_a[index_i], V_a[index_j]

            # Delta_ij, Delta_ji = np.sum(
            #     [(D[i, j] + D[i, k] - D[j, k]) / (2. * (len(D) - 2)) for k in V_a if k != i and k != j]), np.sum(
            #     [(D[i, j] + D[j, k] - D[i, k]) / (2. * (len(D) - 2)) for k in V_a if k != i and k != j])
            # print(i, j, Delta_ij, Delta_ji)

            Delta_ij = (D[i, j] * (len(V_a) - 2) + D[i, V_a].sum() - D[j, V_a].sum()) / (2. * (len(D) - 2))
            Delta_ji = (D[i, j] * (len(V_a) - 2) + D[j, V_a].sum() - D[i, V_a].sum()) / (2. * (len(D) - 2))

            if min(Delta_ij, Delta_ji) < eps:
                pbar.update(1)
                if Delta_ij < Delta_ji:
                    E.append((i, j))
                    V_a.remove(j)
                else:
                    E.append((j, i))
                    V_a.remove(i)
            else:
                pbar.update(2)
                V_a.remove(i)
                V_a.remove(j)
                k = V_a[np.argmin(np.array([abs(D[i, k] + D[j, k] - D[i, j]) for k in V_a]))]
                E += [(k, i), (k, j)]

    if len(V_a) == 2:
        E.append(V_a)

    else:

        ccp, cpc, pcc = abs(D[V_a[0], V_a[2]] + D[V_a[1], V_a[2]] - D[V_a[0], V_a[1]]), abs(
            D[V_a[0], V_a[1]] + D[V_a[2], V_a[1]] - D[V_a[0], V_a[2]]), abs(
            D[V_a[1], V_a[0]] + D[V_a[2], V_a[0]] - D[V_a[1], V_a[2]])
        if ccp < min(cpc, pcc):
            E += [(V_a[2], V_a[0]), (V_a[2], V_a[1])]
        elif cpc < min(ccp, pcc):
            E += [(V_a[1], V_a[0]), (V_a[1], V_a[2])]
        else:
            E += [(V_a[0], V_a[1]), (V_a[0], V_a[2])]

    return (E)

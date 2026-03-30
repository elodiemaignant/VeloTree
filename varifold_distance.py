import numpy as np

def varifold_squared_distance_matrix(curves, sigma_x=1., sigma_t=.1):
    inner = np.zeros((len(curves), len(curves)))

    K_x = lambda mu, nu: np.exp(-np.abs(np.transpose(np.linalg.norm(mu[0], axis=1) ** 2 - 2. * nu[0] @ np.transpose(mu[0])) + np.linalg.norm(nu[0], axis=1) ** 2) / (sigma_x ** 2))
    K_t = lambda mu, nu: np.exp(-np.abs(np.transpose(np.ones(mu[1].shape[0]) - 2. * np.transpose(np.transpose(nu[1]) / np.linalg.norm(nu[1], axis=1)) @ (np.transpose(mu[1]) / np.linalg.norm(mu[1], axis=1))) + np.ones(nu[1].shape[0])) / (sigma_t ** 2))

    varifolds = []
    for curve in curves: 
        if len(curve) < 2:
            curve = np.array([curve[0], curve[0] + sigma_t * np.random.rand(curve.shape[1])])
        varifold = np.stack(((curve[1:] + curve[:-1]) / 2., curve[1:] - curve[:-1]))
        varifolds.append(varifold)

    for i, mu_i in enumerate(varifolds):
        for j, mu_j in enumerate(varifolds[:i+1]):
            inner_ij = np.linalg.norm(mu_i[1], axis=1) @ (K_x(mu_i, mu_j) * K_t(mu_i, mu_j)) @ np.linalg.norm(mu_j[1], axis=1)
            inner[j, i] = inner[i, j] = inner_ij

    return np.diag(inner)[:, None] - 2 * inner + np.diag(inner)[None]

import matplotlib.pyplot as plt
import numpy as np

def smooth_vector_field(point, control_points, control_vectors, sigma):
    k = np.exp(-np.linalg.norm(control_points - point, axis=1) ** 2 / (2. * sigma ** 2))
    weights = k / np.sum(k)
    vector = weights @ control_vectors
    return (vector)

def plot_streamlines(expressions_2D, velocities_2D, sigma=.5):
    n_x = int((max(expressions_2D[:, 0])  - min(expressions_2D[:, 0])) / (sigma / 5.))
    n_y = int((max(expressions_2D[:, 1])  - min(expressions_2D[:, 1])) / (sigma / 5.))
    X, Y = np.meshgrid(np.linspace(min(expressions_2D[:, 0]) - 1., max(expressions_2D[:, 0]) + 1., n_x), 
                       np.linspace(min(expressions_2D[:, 1]) - 1., max(expressions_2D[:, 1]) + 1., n_y))
    
    U, V = np.zeros_like(X), np.zeros_like(Y)
    for i in range(n_y): 
        for j in range(n_x):
            U[i, j], V[i, j] =  smooth_vector_field(np.array([X[i, j], Y[i, j]]), expressions_2D, velocities_2D, sigma=sigma)
    
    start_points = np.copy(expressions_2D)
    for i in range(len(expressions_2D)):
        k = np.argmin((X.flatten() - expressions_2D[i, 0]) ** 2 + (Y.flatten() - expressions_2D[i, 1]) ** 2)
        start_points[i] = np.array([X[k // n_x, k % n_x], Y[k // n_x, k % n_x]])
        U[k // n_x, k % n_x], V[k // n_x, k % n_x] = velocities_2D[i, 0], velocities_2D[i, 1]
    
    speed = np.sqrt(U**2 + V**2)
    
    plt.streamplot(X, Y, U, V, density=5, linewidth=3. * speed / np.max(speed), color='k', maxlength=.11, start_points=expressions_2D, integration_direction='forward')

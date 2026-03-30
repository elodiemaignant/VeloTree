import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

def smoothing_kernel(point, control_points, sigma):
    k = np.exp(-np.linalg.norm(control_points - point, axis=1) ** 2 / (2. * sigma ** 2))
    return (k / np.sum(k))

def trajectory(i, expressions, velocities, sigma=.1, integration_step=.1, margin=2.):
    x = expressions[i].copy()
    grad = -velocities[i].copy()
    traj = [expressions[i].copy()]
    n_it = 0
    while np.min(np.linalg.norm(expressions - x, axis=1)) < margin and n_it < 5000 and np.linalg.norm(grad) > 1E-3:
        x += integration_step * grad
        traj.append(x.copy())
        weights = smoothing_kernel(x, expressions, sigma)
        grad = -weights @ velocities
        n_it += 1
    
    return (np.array(traj))

def trajectories(expressions, velocities, sigma=.1, integration_step=.1, discretization_step=.5, margin=2.):
    trajs = []

    for i in range(0, len(expressions)):

        traj = trajectory(i, expressions, velocities, sigma, integration_step, margin)

        #subsampling
        T, t, t_next = [0], 0, 1
        length = np.sum(np.linalg.norm(traj[:-1] - traj[1:], axis=1))
        
        while t_next < len(traj):
            
            if np.linalg.norm(traj[t_next] - traj[t]) >= discretization_step:
                t = t_next
                t_next +=1
                T.append(t)
            
            else: t_next += 1 
            
        trajs.append(traj[T].astype('double'))

    return trajs

import numpy as np
from skimage.transform import radon


def Radon_Transform(density,x,y,A,theta):
    """
    Computes the radon transform of a 2D gridded function and its derivatives at specified angles.
    Inputs:
        density: 2D density to find Radon transform of (Density2D)
        x: x coordinates of grid (array)
        y: y coordinates of grid (array)
        A: area of each grid cell (float)
        theta: angles of each Radon transform
    Outputs:
        r: Radon transform of 2D density (array)
        dr: Radon transforms of the derivative surfaces (array)
    """
    #if not in matrix form, correct it
    f = density.f.reshape((x.size,y.size))

    #for each of these angles, get the radon projection
    r = radon(f * A,theta=theta,circle=False)

    if type(density.df_dm) == np.ndarray:
        dr = []
        for i in range(density.df_dm.shape[0]):
            df_i = density.df_dm[i,:].reshape((x.size,y.size))
            dr_i = radon(df_i * A,theta=theta,circle=False)
            dr.append(dr_i)
        return r, dr
    else:
        dr = None
        return r,dr

    
def Axes_Setup(x,y,K):
    """
    Sets up the axes and their coordinates for the Radon transform of 2D density functions.
    Inputs:
        x: grid points in x direction (array)
        y: grid points in y direction (array)
        K: the number of projections to make (int)
    Returns:
        s: coordinates along each Radon transform (array)
        theta: the angles used to produce the transforms (array)
        A: the area of each of the 2D cells (float)
    """
    #we need the area of each cell to weight the function
    dx = (x[-1] - x[0]) / (x.size-1)
    dy = (y[-1] - y[0]) / (y.size-1)

    A = dx * dy

    #now we want to pick out slice angles
    theta = np.linspace(0,180,K,endpoint=False)

    #find the centre of the grid
    c = np.array([(x[-1] - x[0]) / 2, (y[-1] - y[0]) / 2])

    l = int(np.ceil(np.sqrt(2) * max((x.size,y.size)))) #length of radon axes

    #we now need to get the correct coordinates along each slice
    pix_dist = np.arange(l) - l/2 #pixels from centre coordinate

    s = np.zeros((l,K)) #initialise coordinate array

    #now loop through slices - can probably do this more efficiently...
    for k in range(K):
        #get scaling for this slice
        scale=np.sqrt((dx*np.cos(np.deg2rad(theta[k])))**2+(dy*np.sin(np.deg2rad(theta[k])))**2)
        
        #find where centre should be projected
        mid = c[0] * np.cos(np.deg2rad(theta[k])) + c[1] * np.sin(np.deg2rad(theta[k]))
        
        #convert pixel distance to true distance
        s[:,k] = pix_dist * scale + mid

    return s, theta, A
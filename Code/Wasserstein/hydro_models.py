import numpy as np
from scipy.stats import expon
from scipy.special import gamma
from scipy.signal import convolve
from scipy.optimize import Bounds, minimize, dual_annealing

class Hydrograph:
    """
    Hydrograph (streamflow time-series) object. 

    Attributes:
        t: times at which streamflow is measured, assumed equally spaced (array)
        q: discharge at corresponding times (array)
        t0: initial time (float)
        T: final time (float)
        N: number of measurements (int)
        dt: time between each measurement (float)
        Q: cumulative discharge (array)
        V: total volume of discharge (float)
        Q_bar: cumulative discharge centred at zero (array)
        cdf: normalised cumulative discharge (array)
    """
    def __init__(self,t,q):
        self.t = t
        self.q = q
        self.t0 = t[0]
        self.T = t[-1]
        self.N = t.size

        self.dt = (self.T - self.t0) / (self.N - 1)
       
        self.Q = self.dt * np.cumsum(q)
        self.V = self.Q[-1]
        
        self.Q_bar = self.Q - self.V/2 #align the medians to be at zero

        self.cdf = self.Q / self.Q[-1]

    def inv_time(self,s):
        """
        Interpolates the locations for a series of CDF values. Note that these CDF values must be sorted from
        smallest to largest before being input.
        Inputs:
            s: series of quantiles (array, must be sorted)
        Outputs:
            t: interpolated times of the quantiles (array)
        """
        t = np.interp(s,self.Q_bar,self.t,left=self.t0,right=self.T)
        return t

    def inv_cdf(self,s):
        t = np.interp(s,self.cdf,self.t)
        return t

class HydroModel:
    def __init__(self,t,func,deriv=None):
        self.t = t
        self.f = func
        self.df = deriv
    
    def __call__(self,m,r):
        # TODO make the output a hydrograph so it can go directly into HW2
        q = self.f(self.t,m,r)
        h = Hydrograph(self.t,q)
        return h

def Wasserstein(f,g,res=100,gamma=1):
    f_c = np.cumsum(f.q)
    g_c = np.cumsum(g.q)

    h = (f_c[-1] - g_c[-1]) **2

    f_c /= f_c[-1]
    g_c /= g_c[-1]

    s = np.linspace(0,1,res)
    ds = s[1] - s[0]
    f_t = np.interp(s,f_c,f.t)
    g_t = np.interp(s,g_c,g.t)

    W2 = ds * np.sum((f_t - g_t)**2)
    return W2 + gamma * h

def Hydro_Wasserstein(f,g,res=100):
    """
    Calculates the hydrograph-Wasserstein distance between two hydrographs.
    Inputs:
        f: source hydrograph (Hydrograph)
        g: target hydrograph (Hydrograph)
        res: number of points to estimate integral with (int)
    Outputs:
        hw2: hydrograph-Wasserstein distance between f and g (float)
    """
    #work out the domain of integration using the larger water mass
    if f.V >= g.V:
        s = np.linspace(f.Q_bar[0],f.Q_bar[-1],res)
    else:
        s = np.linspace(g.Q_bar[0],g.Q_bar[-1],res)
    
    #we can work out the spacing of the points to allow numerical int.
    ds = s[1] - s[0]

    #evaluate both of the inverse functions on the calculated grid
    f_t = f.inv_time(s)
    g_t = g.inv_time(s)

    #now do the numerical integration for the hydrograph-Wasserstein distance
    hw2 = ds * np.sum((f_t - g_t)**2)
    return hw2


def Hydrograph_HW(m,model,obs,r,res=1000,gamma=1):
    """
    Compute the hydrograph-Wasserstein distance for a set of model parameters.
    Inputs:
        m: model parameters for hydrology model (array)
        model: hydrology model (HydroModel)
        obs: observed streamflow (Hydrograph)
        res: resolution of grid for Wasserstein calculation (int)
    Returns:
        hw2: Hydrograph-Wasserstein distance for model parameters (float)
    """
    sim = model(m,r)
    hw2 = Hydro_Wasserstein(sim,obs,res=res)
    return hw2

def Hydrograph_W2(m,model,obs,r,res=1000,gamma=1):
    """
    Compute the 2-Wasserstein distance for a set of model parameters.
    Inputs:
        m: model parameters for hydrology model (array)
        model: hydrology model (HydroModel)
        obs: observed streamflow (Hydrograph)
        res: resolution of grid for Wasserstein calculation (int)
    Returns:
        w2: 2-Wasserstein distance for model parameters (float)
    """
    sim = model(m,r)
    w2 = Wasserstein(sim,obs,res=res,gamma=gamma)
    return w2

def Hydrograph_L2(m,model,obs,r,res=1000,gamma=1):

    #! res and gamma are dummy variables here so it is the same form as Wasserstein
    sim = model(m,r)
    L2 = np.sum((sim.q - obs.q)**2)
    return L2

def Hydrograph_RMSE(m,model,obs,r,res=1000,gamma=1):
    sim = model(m,r)
    rmse = np.sqrt(np.sum((sim.q - obs.q)**2)/sim.N)
    return rmse


def Fit_Hydrograph(obs,r,model,m0,misfit='L2',res=1000,bounds=None,gamma=1,method=None):

    if misfit == 'L2':
        obj = Hydrograph_L2
    if misfit == 'W2':
        obj = Hydrograph_W2
    if misfit == 'HW':
        obj = Hydrograph_HW
    if misfit == 'RMSE':
        obj = Hydrograph_RMSE

    # TODO assign the derivatives in the above if statements and link to scipy

    result = minimize(obj,m0,args=(model,obs,r,res,gamma),bounds=bounds,method=method)
    return result

def Fit_Hydrograph_Global(obs,r,model,bounds,misfit='L2',res=1000,gamma=1):

    if misfit == 'L2':
        obj = Hydrograph_L2
    if misfit == 'W2':
        obj = Hydrograph_W2
    if misfit == 'HW':
        obj = Hydrograph_HW
    if misfit == 'RMSE':
        obj = Hydrograph_RMSE

    # TODO assign the derivatives in the above if statements and link to scipy

    result = dual_annealing(obj,bounds,args=(model,obs,r,res,gamma),maxfun=1e5)
    return result




def HydroBary(hydro_lst,weights=None,res=1000):
    """
    Finds the barycenter of a series of hydrographs.
    Inputs:
        hydro_lst: list of hydrographs to find barycenter of (list of Hydrograph)
        weights: weighting of each hydrograph in barycenter, must add to one (array). If None, 
            equal weighting is assumed.
        res: resultion of inverse cdf grid (int)
    Outputs:
        bary: barycenter of hydrographs (Hydrograph)

    """
    K = len(hydro_lst) #get the number of input densities
    t = hydro_lst[0].t #assume all are on same t grid for now

    if weights == None: #if not given weights, assume all equal
        weights = np.ones(K) / K
  
    #so we first need to get the inverse CDFs on a grid
    s = np.linspace(0,1,res)

    #get the inverse cdf and volume of each hydrograph

    # TODO can we turn this all into one step using loop inside of list?
    icdf = []
    vols = []
    for hydro in hydro_lst:
        icdf.append(hydro.inv_cdf(s))
        vols.append(hydro.V)

    icdf = np.array(icdf) #convert these into arrays
    vols = np.array(vols)

    #take the weighted mean of the inverse cdfs
    bary_icdf = np.sum(icdf * weights[:,None],axis=0)

    #now need to reverse interpolate this to get CDF on desired time grid
    bary_cdf = np.interp(t,bary_icdf,s)

    #remember - my way of getting the CDF is a cumulative sum so just need to
    #undo that to get back to pdf

    cdf_shifted = np.insert(np.delete(bary_cdf, -1), 0, 0)

    pdf = (bary_cdf - cdf_shifted) / hydro_lst[0].dt
    
    #this has unit mass, so need to rescale to have average of hydrograph volumes
    bV = np.sum(weights * vols)
    bq = pdf * bV
    bary = Hydrograph(t,bq)

    return bary


"""
SIMPLE HYDROLOGY MODELS FOR TESTING PURPOSES
"""


def Unit_Hydrograph(t,m,r):
    """
    Apply the instantaneous unit hydrograph define by the given model parameters to the input
    rainfall time-series.
    Inputs:
        t: times at which rainfall was measured (array)
        m: model parameters for unit hydrograph (theta,k,lambda) (array)
        r: rainfall timeseries (array)
    Outputs:
        q: estimated streamflow time-series (array)
    """
    
    u = Gamma(m,t)
    
    q = convolve(u,r)
    q = q[:t.size]
    return q

def Gamma(m,t):
    """
    Calculates a weighted gamma distribution, used as Nash's IUH.
    Inputs:
        m: model parameters (theta,k,lambda) (array)
        t: times at which we want to evaluate the IUH (array)
    Outputs:
        u: unit hydrograph of gamma distribution with input model parameters (array)
    """
    theta = m[0]
    k = m[1]
    
    u = t**(k-1) * np.exp(-t/theta) / (gamma(k)*theta**k)
    return u

def Gamma_Derivs(m,t,h=0.1):
    
    du = np.zeros((m.size,t.size))

    for i in range(m.size):
        add = np.zeros(m.size)
        add[i] = h
        du[i,:] = (Gamma(m+add,t) - Gamma(m-add,t)) / (2*h)

    return du

def Unit_Derivs(t,m,*args):
    """
    Finds the derivative of each point in the hydrograph with respect to the unit hydrograph
    model parameters.
    Inputs:
        t: times at which the streamflow has been measured (array)
        m: model parameters of the unit hydrograph (theta,k,lambda) (array)
        args: rainfall time-series (array)
    Outputs:
        dq: derivative of hydrograph w.r.t. theta,k,lambda (array)
    """
    r = args[0]
    du = Gamma_Derivs(m,t)
    
    dq = np.zeros((3,t.size))
    
    for i in range(3):
        dq[i,:] = convolve(du[i,:],r)[:t.size]
        
    return dq

def Random_Rain(t_max,T,lamb,alpha,beta):
    """
    Generates a random synthetic rainfall time-series based upon a Poisson process.
    """
    t = 0
    storms = []
    while t < t_max:
        t += expon.rvs(scale=1/lamb)
        storms.append(t)
    storms = np.array(storms[:-1])
    length = expon.rvs(scale=1/alpha,size=storms.size)
    intensity = expon.rvs(scale=1/beta,size=storms.size)
    storm_end = storms + length

    times = np.linspace(0,t_max,T)
    rain = np.zeros(T)
    for i in range(storms.size):
        s_i = storms[i]
        e_i = storm_end[i]
        ind = np.where((s_i < times) & (times < e_i))
        rain[ind] += intensity[i]
        
    return rain


def Random_Rain_Noise(t_max,T,lamb,alpha,beta,s_noise=1,l_noise=1,i_noise=1):
    """
    Generates two rainfall time-series: one original random (true rainfall) and one corrupted by
    amplitude and timing errors (measured rainfall). Aims to simulate the measurement errors
    in rainfall measurement.
    
    """
    t = 0
    storms = []
    while t < t_max:
        t += expon.rvs(scale=1/lamb)
        storms.append(t)
    storms = np.array(storms[:-1])
    storms_n = storms + s_noise * np.random.randn(storms.size)
    
    
    length = expon.rvs(scale=1/alpha,size=storms.size)
    length_n = length + l_noise * np.random.randn(length.size)
    
    intensity = expon.rvs(scale=1/beta,size=storms.size)
    intensity_n = intensity + i_noise * np.random.randn(length.size)
    
    storm_end = storms + length
    storm_end_n = storms_n + length_n
    
    times = np.linspace(0,t_max,T)
    rain = np.zeros(T)
    rain_n = np.zeros(T)
    
    for i in range(storms.size):
        s_i = storms[i]
        e_i = storm_end[i]
        ind = np.where((s_i < times) & (times < e_i))
        rain[ind] += intensity[i]
        
        s_i_n = storms_n[i]
        e_i_n = storm_end_n[i]
        ind_n = np.where((s_i_n < times) & (times < e_i_n))
        rain_n[ind_n] += intensity_n[i]
        
    return rain, rain_n

def IUH_Bounds():
    lb = np.array([0,1,0])
    ub = np.array([np.inf,np.inf,1])
    bounds = Bounds(lb,ub)
    return bounds



def NonLinearHydro(t,m,r,V=0):
    V = 0

    q = np.zeros(t.size)

    for i,t in enumerate(t):
        q[i] = (V / m[0])**m[1]
        dV = m[2] * r[i] - q[i]
        V += dV
    return q


def Simple_Rain(tn,n_event,ev_length=5,t_err=3):
    r = np.zeros(tn)
    r_n = np.zeros(tn)
    
    t_start = np.random.randint(low=t_err,high=tn-1-ev_length-t_err,size=n_event)
    t_length = np.random.randint(low=1,high=ev_length,size=n_event)
    inten = expon.rvs(size=n_event)
    
    t_err = np.random.randint(low=-t_err,high=t_err,size=n_event)
    
    for i in range(n_event):
        r[t_start[i]:t_start[i]+t_length[i]] = inten[i]
        r_n[t_start[i]+t_err[i]:t_start[i]+t_err[i]+t_length[i]] = inten[i]
    
    return r, r_n
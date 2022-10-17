import numpy as np
from scipy.optimize import minimize
from scipy import interpolate
#from .radon_t import Radon_Transform, Axes_Setup

class Density1D:
    def __init__(self,x,f,df_dm=None,dx=None):

        self.f = f
        self.x = x
        self.N = x.size

        if dx == None:
            #if not given cell sizes, calculate from min and max
            self.dx = (x[-1] - x[0]) / (self.N - 1)
        else:
            self.dx = dx

        F = self.dx * np.cumsum(f)
        self.V = F[-1]
        self.F = F / self.V #normalise the sum to 1
        self.f_hat = f / self.V #normalise function
  
        if type(df_dm) == np.ndarray:
            #want to convert model derivatives to being of normalised density function
            self.df_dm = df_dm
            self.df_hat_dm = (df_dm * self.V - (self.f[:,None]  * self.dx * np.sum(df_dm,axis=1)).T) / self.V**2

    def inv_cdf(self,t):
        """
        Interpolates the locations for a series of CDF values. Note that these CDF values must be sorted from
        smallest to largest before being input.
        Inputs:
            t: series of quantiles (array, must be sorted)
        Outputs:
            x: interpolated locations of the quantiles (array)
        """
        x = np.interp(t,self.F,self.x)
        return x



class Model1D:
    def __init__(self,x,func,deriv=None,dx=None):
        """
        Creates a forward model object, which can then produce density functions for given model
        parameters.
        Inputs:
            func: the forward model, takes a set of evaluation points and model parameters (function)
            deriv: derivatives of the forward model with respect to model parameters, takes evaluations
            points and model parameters (function)
            x: points at which the forward model will be evaluated (numpy array)
        """
        self.f = func
        self.df = deriv
        self.x = x
        self.dx = dx

    def __call__(self,m,*args):
        """
        Solves the forward problem for a given set of model parameters.
        """
        return self.f(self.x,m,args)

    def deriv(self,m,*args):
        """
        Find the derivative of each point with respect to each model parameter.
        Inputs:
            m: model parameters
            args: additional arguments for forward model
        """
        if self.df == None:
            return self.finite_diff(m,args)
        else:
            return self.df(self.x,m,args)
    
    def density(self,m,*args):
        """
        Creates a 1D density function for the forward model output.
        Inputs:
            m: the model parameters to use for the forward model run (array)
        Outputs:
            dens: density object with information from forward model (Density1D)
        """
        f = self.f(self.x,m,*args)

        if self.df == None:
            df = self.finite_diff(m,*args)
        else:
            df = self.df(self.x,m,*args)
        dens = Density1D(self.x,f,df_dm=df,dx=self.dx)
        return dens

    def finite_diff(self,m,*args,h=0.01):

        df = np.zeros((m.size,self.x.size))

        for i in range(m.size):
            add = np.zeros(m.size)
            add[i] = h
            df[i,:] = (self.f(self.x,m + add,*args) - self.f(self.x,m - add,*args)) / (2*h)

        return df

def Wasserstein(m,model,target):
    """
    Computes the 2-Wasserstein distance between the target density and
    forward modelled density.
    Inputs:
        m: model parameters (array)
        model: forward model (Model1D)
        target: observed data as density function (Density1D)
    Outputs:
        W2: 2-Wasserstein distance for model parameters (float)
    """
    source = model.density(m)
    x = source.x
    Tx = target.inv_cdf(source.F)

    W2 = source.dx * np.sum((x - Tx)**2 * source.f_hat)

    return W2


def dWasserstein(m,model,target):
    """
    Derivative of Wasserstein distance with respect to each model parameter.
    Inputs:
        m: model parameters (array)
        model: forward model (Model1D)
        target: observed data as density function (Density1D)
    Outputs:
        dW: derivative of W2 w.r.t. each parameter in m (array)
    """
    source = model.density(m)
    x = source.x
    Tx = target.inv_cdf(source.F)

    dF_dm = source.dx * np.cumsum(source.df_hat_dm,axis=1)
    dW = - 2 * source.dx * np.sum(dF_dm * (x - Tx),axis=1)

    return dW

def L2(m,model,target,*args):
    """
    Compute the L2 misfit between a model output and observations.
    Inputs:
        m: model parameters (array)
        model: forward model (Model1D or ModelND)
        target: observations (Density1D or DensityND)
    Outputs:
        misfit: L2 misfit of simulation and observations (float)
    """
    source = model.density(m,*args)
    misfit = np.sum((source.f - target.f)**2)
    return misfit


def dL2(m,model,target,*args):
    """
    Compute the derivative of the L2 misfit with respect to each
    model parameter.
    Inputs:
        m: model parameters (array)
        model: forward model (Model1D or ModelND)
        target: observations (Density1D or DensityND)
    Outputs:
        dL: derivative of L2 misfit with respect to each component (array)
    """
    source = model.density(m,*args)
    dL = 2 * np.sum((source.f - target.f) * source.df_dm,axis=1)
    return dL


def Invert(model,m0,target):
    """
    Solve inverse problem by minimising 1D Wasserstein distance between source and
    target density functions.
    Inputs:
        model: forward model (Model1D)
        m0: initial estimate of model parameters (array)
        target: density function of observations (Density1D)
    Outputs: 
        fit: optimisation result (scipy.optimize.OptimizeResult)
    """
    fit = minimize(Wasserstein,m0,jac=dWasserstein,args=(model,target),method='BFGS')
    return fit


def Objective(m,model,target,*args,gamma=1,penalty='log'):
    """
    Computes the objective function for mass-variable Wasserstein inversion.
    This takes into account both differences in shape (Wasserstein distance) and 
    total mass (penalty function).
    Inputs:
        m: model parameters (array)
        model: forward model (Model1D)
        target: observed data as density function (Density1D)
        gamma: weighting of the penalty term (float, default = 1)
    Outputs:
        J: Objective function for model parameters (float)
    """
    source = model.density(m,*args)
    x = source.x
    Tx = target.inv_cdf(source.F)

    Wp = source.dx * np.sum((x - Tx)**2 * source.f_hat)

    h = 0

    if penalty == 'log':
        h = np.log(source.V / target.V)**2
    
    if penalty == 'diff':
        h = (source.V - target.V)**2

    J = Wp + gamma * h
    return J

def dObjective(m,model,target,*args,gamma=1):
    """
    Derivative of mass-dependent Wasserstein distance with respect to each 
    model parameter.
    Inputs:
        m: model parameters (array)
        model: forward model (Model1D)
        target: observed data as density function (Density1D)
        gamma: weighting of the penalty term (float, default = 1)
    Outputs:
        dJ: derivative of objective function w.r.t. each parameter in m (array)
    """
    source = model.density(m,*args)
    x = source.x
    Tx = target.inv_cdf(source.F)

    dF_dm = source.dx * np.cumsum(source.df_hat_dm,axis=1)
    dW = - 2 * source.dx *  np.sum(dF_dm * (x - Tx),axis=1)

    dh = 2 * source.dx * (np.log(source.V) - np.log(target.V)) * (np.sum(source.df_dm,axis=1) / source.V)
    dJ = dW + gamma * dh

    return dJ


def Invert_Scale(model,m0,target,*args,gamma=1,bounds=None):
    """
    Solve inverse problem by minimising 1D mass-dependent Wasserstein distance between 
    source and target density functions.
    Inputs:
        model: forward model (Model1D)
        m0: initial estimate of model parameters (array)
        target: density function of observations (Density1D)
        gamma: weighting of the penalty term (float, default = 1)
    Outputs: 
        fit: optimisation result (scipy.optimize.OptimizeResult)
    """
    fit = minimize(Objective,m0,jac=dObjective,args=(model,target,*args,gamma),bounds=bounds)
    return fit


def Barycentre1D(dens_lst,w=None,res=1000):
    """
    Compute the barycentre of a series of 1D densities.
    Inputs:
        dens_lst: densities to find barycentre of (list of Density1D)
        w: contribution of each density to barycentre, must sum to one (array)
        res: number of interpolation grid points for inverse CDF (int)
    Outputs:
        bary: barycentre of input densities (Density1D) 
    """


    K = len(dens_lst) #get the number of input densities
    x = dens_lst[0].x #assume all are on same x grid for now

    if w == None: #if not given weights, assume all equal
        w = np.ones(K) / K
  
    #so we first need to get the inverse CDFs on a grid
    t = np.linspace(0,1,res)

    icdf = []
    for dens in dens_lst:
        icdf.append(dens.inv_cdf(t))
    icdf = np.array(icdf)

    #take the weighted mean of the inverse cdfs
    bary_icdf = np.sum(icdf * w[:,None],axis=0)

    #now need to reverse interpolate this to get CDF on desired x grid
    bary_cdf = np.interp(x,bary_icdf,t)

    #remember - my way of getting the CDF is a cumulative sum so just need to
    #undo that to get back to pdf

    cdf_shifted = np.insert(np.delete(bary_cdf, -1), 0, 0)

    pdf = (bary_cdf - cdf_shifted) / dens_lst[0].dx

    bary = Density1D(x,pdf)
    return bary


""" 
--------------------------------------------------------------
2D stuff from here - not currently in use
--------------------------------------------------------------
"""


# class Density2D:
#     def __init__(self,x,y,f,df_dm=None,A=None):
#         self.x = x
#         self.y = y
#         self.N = x.size * y.size
#         self.f = f
#         self.df_dm = df_dm
        
#         if A == None:
#             dx = (x[-1] - x[0]) / (x.size - 1)
#             dy = (y[-1] - y[0]) / (y.size - 1)
#             self.A = dx * dy
#         else:
#             self.A = A

#         self.V = np.sum(self.f * self.A)
#         self.f_hat = f / self.V
  
#         if type(self.df_dm) == np.ndarray:
#             self.df_hat_dm = (df_dm * self.V - (self.f[:,None]  * self.A * np.sum(df_dm,axis=-1)).T) / self.V**2

# class Model2D:
#     def __init__(self,x,y,func,deriv=None):
#         """
#         Creates a forward model object, which can then produce density functions for given model
#         parameters.
#         Inputs:
#             func: the forward model, takes a set of evaluation points and model parameters (function)
#             deriv: derivatives of the forward model with respect to model parameters, takes evaluations
#             points and model parameters (function)
#             x: points at which the forward model will be evaluated (numpy array)
#         """
#         self.x = x
#         self.y = y
#         X,Y = np.meshgrid(x,y)
#         xy = np.column_stack((X.flatten(),Y.flatten())).T
#         self.N = x.size * y.size

#         dx = (x[-1] - x[0]) / (x.size-1)
#         dy = (y[-1] - y[0]) / (y.size-1)
#         self.f = func
#         self.df = deriv
#         self.xy = xy
#         self.A = dx * dy

#     def __call__(self,m,*args):
#         """
#         Solves the forward problem for a given set of model parameters.
#         """
#         return self.f(self.xy,m,args)

#     def deriv(self,m,*args):
#         if self.df == None:
#             return self.finite_diff(m,args)
#         else:
#             return self.df(self.xy,m,args)
    
#     def density(self,m,*args):
#         """
#         Creates a ND density function for the forward model output.
#         Inputs:
#             m: the model parameters to use for the forward model run (array)
#         Outputs:
#             dens: density object with information from forward model (Density1D)
#         """
#         f = self.f(self.xy,m,*args)

#         if self.df == None:
#             df = self.finite_diff(m,*args)
#         else:
#             df = self.df(self.xy,m,*args)
#         dens = Density2D(self.x,self.y,f,df_dm=df,A=self.A)
#         return dens

#     def finite_diff(self,m,*args,h=0.01):

#         df = np.zeros((m.size,self.N))

#         for i in range(m.size):
#             add = np.zeros(m.size)
#             add[i] = h
#             df[i,:] = (self.f(self.xy,m + add,*args) - self.f(self.xy,m - add,*args)) / (2*h)

#         return df


# def Radon_Wasserstein(m,model,targ_lst,proj_set):
#     """
#     Calculates the Radon Wasserstein distance between a given model and
#     the target observations, expressed as a set of 1D projections
#     Inputs:
#         m: model parameters (array)
#         model: forward model (ModelND)
#         targ_lst: the density of each Radon projection (list of Density1D)
#         proj_set: projections of observations (Projection_Set)
#     Outputs:
#         sw2: squared Radon 2-Wasserstein distance (float)
#     """

#     #run the forward model to get a source density function
#     source = model.density(m)

#     source_lst = proj_set.Transform(source)

#     sw2 = 0 #initialise sliced Wasserstein

#     #now loop through each projection and get 1D distance
#     for k in range(proj_set.K):
        
#         #find source and target pairs
#         x = source_lst[k].x
#         Tx = targ_lst[k].inv_cdf(source_lst[k].F)

#         #find the 1D distance
#         W2 = source_lst[k].dx * np.sum((x - Tx)**2 * source_lst[k].f_hat)
#         sw2 += W2

#     sw2 /= proj_set.K #take mean to get sliced Wasserstein
#     return sw2

# def dRadon_Wasserstein(m,model,targ_lst,proj_set):
#     """
    
#     """
#     source = model.density(m)

#     source_lst = proj_set.Transform(source)
    
#     dsw = np.zeros(m.size)

#     #now loop through each projection and get 1D distance
#     for k in range(proj_set.K):

#         #find source and target pairs
#         x = source_lst[k].x
#         Tx = targ_lst[k].inv_cdf(source_lst[k].F)

#         dF_dm = source_lst[k].dx * np.cumsum(source_lst[k].df_hat_dm,axis=1)
#         dW = - 2 * source_lst[k].dx * np.sum(dF_dm * (x - Tx),axis=1)

#         dsw += dW
#     dsw /= proj_set.K
#     return dsw


# class Projection_Set:
#     def __init__(self,x,y,K=30):
#         #want to make this set up the axes and their coordinate. Then has method where
#         #we can pass any density through it and it populates those axes with radon values
#         self.K = K
#         s, theta, A = Axes_Setup(x,y,K)

#         self.s = s
#         self.theta = theta
#         self.A = A
#         self.x = x
#         self.y = y

#     def Transform(self,density):
#         """
#         Computes the Radon transforms of a density for this projection set.
#         Inputs:
#             density: the 2D density function for which we want Radon transforms.
#         Outputs:
#             slices: each of the 1D Radon transform densities (list of Density1D)
#         """
#         slices = [] #initialise list for 1D density functions

#         #we can pass our density function to the Radon calculator
#         r, dr = Radon_Transform(density,self.x,self.y,self.A,self.theta)

#         #we now want to unpack this result along each of the slices
#         for k in range(self.K):

#             #we can immediately get the density function from the output matrix
#             f_k = r[:,k]

#             #if we have derivatives, we need to unpack these
#             if type(dr) == list:
#                 df_k = []
#                 for i in range(len(dr)): #loop through each model parameter
#                     dr_i = dr[i] #get the radon matrix
#                     df_k.append(dr_i[:,k]) #pull out the desired column
#                 df_k = np.array(df_k) #put these back together into a matrix
#                 dens_k = Density1D(self.s[:,k],f_k,df_dm=df_k) #combine into density
#             else: #no derivatives, so straight to density
#                 dens_k = Density1D(self.s[:,k],f_k)
#             slices.append(dens_k)
#         return slices


# def Invert2D(model,m0,target,K=10):
#     """
#     Solve inverse problem by minimising the Radon Wasserstein distance between source and
#     target density functions.
#     Inputs:
#         model: forward model (Model2D)
#         m0: initial estimate of model parameters (array)
#         target: density function of observations (Density2D)
#     Outputs: 
#         fit: optimisation result (scipy.optimize.OptimizeResult)
#     """
#     proj_set = Projection_Set(model.x,model.y,K) #firstly generate the set of projection axes that we will need
#     targ_lst = proj_set.Transform(target) #now take the Radon transforms of target - these won't change

#     #now we can perform the optimisation - can change method if bounds are need 
#     fit = minimize(Radon_Wasserstein,m0,jac=dRadon_Wasserstein,args=(model,targ_lst,proj_set))
#     return fit



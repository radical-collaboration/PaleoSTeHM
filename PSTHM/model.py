#----------------------Define Functions---------------------------

import numpy as np
import torch
import torch.distributions as torchdist
from torch.distributions import constraints
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.contrib.gp.models.model import GPModel
from pyro.contrib.gp.util import conditional
from pyro.nn.module import PyroParam, pyro_method
from pyro.util import warn_if_nan
from scipy import interpolate
from tqdm.notebook import tqdm
import torch
from torch.distributions import constraints
from pyro.contrib.gp.kernels.kernel import Kernel

#----------------------GP models---------------------------

class GPRegression_V(GPModel):
    r"""
    Gaussian Process Regression model.

    The core of a Gaussian Process is a covariance function :math:`k` which governs
    the similarity between input points. Given :math:`k`, we can establish a
    distribution over functions :math:`f` by a multivarite normal distribution

    .. math:: p(f(X)) = \mathcal{N}(0, k(X, X)),

    where :math:`X` is any set of input points and :math:`k(X, X)` is a covariance
    matrix whose entries are outputs :math:`k(x, z)` of :math:`k` over input pairs
    :math:`(x, z)`. This distribution is usually denoted by

    .. math:: f \sim \mathcal{GP}(0, k).

    .. note:: Generally, beside a covariance matrix :math:`k`, a Gaussian Process can
        also be specified by a mean function :math:`m` (which is a zero-value function
        by default). In that case, its distribution will be

        .. math:: p(f(X)) = \mathcal{N}(m(X), k(X, X)).

    Given inputs :math:`X` and their noisy observations :math:`y`, the Gaussian Process
    Regression model takes the form

    .. math::
        f &\sim \mathcal{GP}(0, k(X, X)),\\
        y & \sim f + \epsilon,

    where :math:`\epsilon` is Gaussian noise.

    .. note:: This model has :math:`\mathcal{O}(N^3)` complexity for training,
        :math:`\mathcal{O}(N^3)` complexity for testing. Here, :math:`N` is the number
        of train inputs.

    Reference:

    [1] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    :param torch.Tensor X: A input data for training. Its first dimension is the number
        of data points.
    :param torch.Tensor y: An output data for training. Its last dimension is the
        number of data points.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object, which
        is the covariance function :math:`k`.
    :param torch.Tensor noise: Variance of Gaussian noise of this model.
    :param callable mean_function: An optional mean function :math:`m` of this Gaussian
        process. By default, we use zero mean.
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    """

    def __init__(self, X, y, kernel, noise=None, mean_function=None, jitter=1e-6):
        assert isinstance(
            X, torch.Tensor
        ), "X needs to be a torch Tensor instead of a {}".format(type(X))
        if y is not None:
            assert isinstance(
                y, torch.Tensor
            ), "y needs to be a torch Tensor instead of a {}".format(type(y))
        super().__init__(X, y, kernel, mean_function, jitter)

#         noise = self.X.new_tensor(1.0) if noise is None else noise
        self = self.double() #GP in pyro should use double precision
        self.X = self.X.double()
        self.y = self.y.double()
        if noise is None:
            noise = self.X.new_tensor(1.0)
            self.noise = PyroParam(noise, constraints.positive)
        else:
            if  noise.dim() <=1:
                noise_store = torch.zeros(len(self.X),len(self.X))
                noise_store.view(-1)[:: len(self.X) + 1] += noise 
                self.noise = noise_store.double()
            elif noise.dim() ==2:
                self.noise = noise.double()
               
    @pyro_method
    def model(self):
        self.set_mode("model")

        N = self.X.size(0)
        Kff = self.kernel(self.X)
        if self.noise.dim() <=1:
            Kff.view(-1)[:: N + 1] += self.jitter + self.noise
        elif self.noise.dim() ==2:
            Kff = Kff + self.noise
            Kff.view(-1)[:: N + 1] += self.jitter
        Lff = torch.linalg.cholesky(Kff)

        zero_loc = self.X.new_zeros(self.X.size(0))
        f_loc = zero_loc + self.mean_function(self.X)
        if self.y is None:
            f_var = Lff.pow(2).sum(dim=-1)
            return f_loc, f_var
        else:
            return pyro.sample(
                self._pyro_get_fullname("y"),
                dist.MultivariateNormal(f_loc, scale_tril=Lff)
                .expand_by(self.y.shape[:-1])
                .to_event(self.y.dim() - 1),
                obs=self.y,
            )


    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()


    def forward(self, Xnew, full_cov=False):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, \epsilon) = \mathcal{N}(loc, cov).

        .. note:: The noise parameter ``noise`` (:math:`\epsilon`) together with
            kernel's parameters have been learned from a training procedure (MCMC or
            SVI).

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :param bool noiseless: A flag to decide if we want to include noise in the
            prediction output or notest_cov[i:i+1]t.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")

        N = self.X.size(0)
        Kff = self.kernel(self.X).contiguous()
        if self.noise.dim() <=1:
            Kff.view(-1)[:: N + 1] += self.jitter + self.noise
        elif self.noise.dim() ==2:
            Kff = Kff + self.noise
            Kff.view(-1)[:: N + 1] += self.jitter
        # Kff.view(-1)[:: N + 1] += self.jitter + self.noise  # add noise to the diagonal
        Lff = torch.linalg.cholesky(Kff)

        y_residual = self.y - self.mean_function(self.X)
        loc, cov = conditional(
            Xnew,
            self.X,
            self.kernel,
            y_residual,
            None,
            Lff,
            full_cov,
            jitter=self.jitter,
        )

        return loc + self.mean_function(Xnew), cov


class GPRegression_EIV(GPModel):
    r"""
    Gaussian Process Regression model.

    The core of a Gaussian Process is a covariance function :math:`k` which governs
    the similarity between input points. Given :math:`k`, we can establish a
    distribution over functions :math:`f` by a multivarite normal distribution

    .. math:: p(f(X)) = \mathcal{N}(0, k(X, X)),

    where :math:`X` is any set of input points and :math:`k(X, X)` is a covariance
    matrix whose entries are outputs :math:`k(x, z)` of :math:`k` over input pairs
    :math:`(x, z)`. This distribution is usually denoted by

    .. math:: f \sim \mathcal{GP}(0, k).

    .. note:: Generally, beside a covariance matrix :math:`k`, a Gaussian Process can
        also be specified by a mean function :math:`m` (which is a zero-value function
        by default). In that case, its distribution will be

        .. math:: p(f(X)) = \mathcal{N}(m(X), k(X, X)).

    Given inputs :math:`X` and their noisy observations :math:`y`, the Gaussian Process
    Regression model takes the form

    .. math::
        f &\sim \mathcal{GP}(0, k(X, X)),\\
        y & \sim f + \epsilon,

    where :math:`\epsilon` is Gaussian noise.

    .. note:: This model has :math:`\mathcal{O}(N^3)` complexity for training,
        :math:`\mathcal{O}(N^3)` complexity for testing. Here, :math:`N` is the number
        of train inputs.

    Reference:

    [1] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    :param torch.Tensor X: A input data for training. Its first dimension is the number
        of data points.
    :param torch.Tensor y: An output data for training. Its last dimension is the
        number of data points.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object, which
        is the covariance function :math:`k`.
    :param x_noise: A input data for training. Its first dimension is the number, representing 
        the variance of the error for age, which is usually obtained from radiocarbon dating 
    :param torch.Tensor noise: Variance of Gaussian noise of this model.
    :param callable mean_function: An optional mean function :math:`m` of this Gaussian
        process. By default, we use zero mean.
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    """

    def __init__(self, X, y, x_noise,kernel, noise=None, mean_function=None, jitter=1e-6):
        assert isinstance(
            X, torch.Tensor
        ), "X needs to be a torch Tensor instead of a {}".format(type(X))
        if y is not None:
            assert isinstance(
                y, torch.Tensor
            ), "y needs to be a torch Tensor instead of a {}".format(type(y))
       
        super().__init__(X, y,kernel, mean_function, jitter)
       
        if len(torch.tensor(x_noise).shape)==0:
            x_noise = torch.ones(len(X))*x_noise
        self.x_noise = x_noise.double()
        self = self.double() #GP in pyro should use double precision
        self.X = self.X.double()
        self.y = self.y.double()

        if noise is None:
            noise = self.X.new_tensor(1.0)
            self.noise = PyroParam(noise, constraints.positive)
        else:
            if  noise.dim() <=1:
                noise_store = torch.zeros(len(self.X),len(self.X))
                noise_store.view(-1)[:: len(self.X) + 1] += noise 
                self.noise = noise_store.double()
            elif noise.dim() ==2:
                self.noise = noise.double()
        
    @pyro_method
    def model(self):
        self.set_mode("model")
        N = self.X.size(0)
        x_noise = pyro.sample('xerr',dist.Normal(torch.zeros(N),self.x_noise**0.5).to_event(1))
        if self.X.dim()<=1:
            X_noisy = self.X
            X_noisy = (self.X+x_noise)
        else:
            X_noisy = torch.clone(self.X)
            X_noisy[:,0] += x_noise
       
        Kff = self.kernel(X_noisy).contiguous()
        Kff = Kff + self.noise
        Kff.view(-1)[:: N + 1] += self.jitter
        # Kff.view(-1)[:: N + 1] += self.jitter + self.noise  # add noise to diagonal
        Lff = torch.linalg.cholesky(Kff)
        zero_loc = X_noisy.new_zeros(X_noisy.size(0))
        f_loc = zero_loc + self.mean_function(X_noisy)
        if self.y is None:
            f_var = Lff.pow(2).sum(dim=-1)
            return f_loc, f_var
        else:

            return pyro.sample(
                self._pyro_get_fullname("y"),
                dist.MultivariateNormal(f_loc, scale_tril=Lff)
                .expand_by(self.y.shape[:-1])
                .to_event(self.y.dim() - 1),
                obs=self.y,
            )


    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()

    def forward(self, Xnew, full_cov=False, noiseless=True):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, \epsilon) = \mathcal{N}(loc, cov).

        .. note:: The noise parameter ``noise`` (:math:`\epsilon`) together with
            kernel's parameters have been learned from a training procedure (MCMC or
            SVI).

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :param bool noiseless: A flag to decide if we want to include noise in the
            prediction output or not.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")

        N = self.X.size(0)
        Kff = self.kernel(self.X).contiguous()
        if self.noise.dim() <=1:
            Kff.view(-1)[:: N + 1] += self.jitter + self.noise
        elif self.noise.dim() ==2:
            Kff = Kff + self.noise
            Kff.view(-1)[:: N + 1] += self.jitter

        # Kff.view(-1)[:: N + 1] += self.jitter + self.noise  # add noise to the diagonal
        Lff = torch.linalg.cholesky(Kff)

        y_residual = self.y - self.mean_function(self.X)
        loc, cov = conditional(
            Xnew,
            self.X,
            self.kernel,
            y_residual,
            None,
            Lff,
            full_cov,
            jitter=self.jitter,
        )

        if full_cov and not noiseless:
            M = Xnew.size(0)
            cov = cov.contiguous()
            cov.view(-1, M * M)[:, :: M + 1] += self.noise  # add noise to the diagonal
        if not full_cov and not noiseless:
            cov = cov + self.noise.abs()

        return loc + self.mean_function(Xnew), cov

class GIA_ensemble(Kernel):
    '''
    This is a class to define a GIA ensemble model as the mean function for GP

    ------------Inputs--------------
    GIA_model_interp: a list of interpolation function that can 3D interpolate the
    RSL history predicted by a GIA model

    ------------Outputs--------------
    mean: the prediction of the GIA ensemble model
    '''
    def __init__(self,GIA_model_interp,input_dim=1):
        super().__init__(input_dim)
        self.GIA_model_interp = GIA_model_interp
        self.GIA_model_num =len(GIA_model_interp)
        self.s = PyroParam(torch.tensor(1.0))
        self.w = PyroParam(torch.ones(self.GIA_model_num))
       
    def forward(self, X):
        pred_matrix = torch.ones(self.GIA_model_num,X.shape[0])
        for i in range(self.GIA_model_num):
            pred_matrix[i] = torch.tensor(self.GIA_model_interp[i](X.detach().numpy()))
        return ((self.w*self.s)[:,None] *pred_matrix).sum(axis=0)

#----------------------Linear regression models---------------------------

def change_point_model(X, y,x_sigma,y_sigma,n_cp,intercept_prior,coefficient_prior):
    '''
    A function to define a change-point model in pyro

    ------------Inputs--------------
    X: 2D torch tensor with shape (n_samples,n_features)
    y: 1D torch tensor with shape (n_samples)
    x_sigma: float, standard deviation of the error for age, which is obtained from the age data model
    y_sigma: float, standard deviation or covariance function of the error for the RSL, which is obtained from the RSL data model
    n_cp: int, number of change-points
    intercept_prior: pyro distribution for the intercept coefficient
    coefficient_prior: pyro distribution for the slope coefficient

    '''
    # Define our intercept prior
    b = pyro.sample("b", intercept_prior)
    beta_coef_list = torch.zeros(n_cp+1)
    cp_loc_list = torch.zeros(n_cp)
    #Define our coefficient prior
    cp_loc_prior = torch.linspace(X[:,0].min(),X[:,0].max(),n_cp+3)[1:-1]
    gap = cp_loc_prior[1]-cp_loc_prior[0]
    initial_age, ending_age = X[:,0].min()-gap,X[:,0].max()+gap
    
    for i in range(n_cp+1):
        beta_coef = pyro.sample(f"a_{i}", coefficient_prior)    
        beta_coef_list[i] = beta_coef
        if i<n_cp:
            cp_prior = dist.Uniform(cp_loc_prior[i]-gap,cp_loc_prior[i+1]+gap)
            cp_loc = pyro.sample(f"cp_{i}", cp_prior)
            cp_loc_list[i] = cp_loc

    x_noise = pyro.sample('xerr',dist.Normal(0,x_sigma).to_event(1))
    x_noisy = X[:, 0]+x_noise
    mean = torch.zeros(X.shape[0])
    last_intercept = b
    for i in range(n_cp+1):
        if i==0:
            start_age = initial_age
            start_idx = 0
            end_age = cp_loc_list[i]
            try:
                end_idx = torch.where(x_noisy<end_age)[0][-1]+1
            except:
                end_idx = 0
            
            last_change_point = start_age
        elif i==n_cp:
            start_age = cp_loc_list[i-1]
            try:
                start_idx = torch.where(x_noisy>=start_age)[0][0]
            except:
                start_idx = len(X[:, 0])-1
            end_age = ending_age
            end_idx = X.shape[0]
        else:
            start_age = cp_loc_list[i-1]
            start_idx = torch.where(x_noisy>=start_age)[0][0]
            end_age = cp_loc_list[i]
            end_idx = torch.where(x_noisy<end_age)[0][-1]+1

        mean[start_idx:end_idx] = beta_coef_list[i] * (x_noisy[start_idx:end_idx]-last_change_point) + last_intercept
        last_intercept = beta_coef_list[i] * (end_age-last_change_point) + last_intercept
        last_change_point = end_age
        
    with pyro.plate("data", y.shape[0]):        
        # Condition the expected mean on the observed target y
        if y_sigma.dim() <=1:
            pyro.sample("obs", dist.Normal(mean, y_sigma), obs=y)
        elif y_sigma.dim() ==2:
            pyro.sample("obs", dist.MultivariateNormal(mean, y_sigma), obs=y)

def ensemble_GIA_model(X, y,x_sigma,y_sigma,model_ensemble,model_age):
    '''
    A function to define a linear model in pyro

    ------------Inputs--------------
    X: 2D torch tensor with shape (n_samples,n_features)
    y: 1D torch tensor with shape (n_samples)
    x_sigma: float, standard deviation of the error for age, which is obtained from the age data model
    y_sigma: float, standard deviation or covariance function of the error for the RSL, which is obtained from the RSL data model

    '''
    # Define our intercept prior
    # weight_facor_list = torch.zeros(model_ensemble.shape[0])
    # for i in range(model_ensemble.shape[0]):
    weight_facor_list = pyro.sample("W",dist.Dirichlet(torch.ones(model_ensemble.shape[0])))
    # weight_facor_list[i] = weight_factor
    #generate random error for age
    x_noise = pyro.sample('obs_xerr',dist.Normal(0,x_sigma).to_event(1))
    x_noisy = X[:, 0]+x_noise
    #interpolate GIA model
    mean = torch.zeros(len(x_noisy))
    for i in range(model_ensemble.shape[0]):
        GIA_model = interpolate.interp1d(model_age.detach().numpy(),model_ensemble[i].detach().numpy())
        x_noisy[x_noisy>X[:, 0].max()] = X[:, 0].max()
        x_noisy[x_noisy<X[:, 0].min()] = X[:, 0].min()

        #calculate mean prediction
       
        mean += torch.tensor(GIA_model(x_noisy.detach().numpy())) *weight_facor_list[i]
   
    with pyro.plate("data", y.shape[0]):        
        # Condition the expected mean on the observed target y
        if y_sigma.dim() <=1:
            pyro.sample("obs", dist.Normal(mean, y_sigma), obs=y)
        elif y_sigma.dim() ==2:
            pyro.sample("obs", dist.MultivariateNormal(mean, y_sigma), obs=y)
   

def linear_model(X, y,x_sigma,y_sigma,intercept_prior,coefficient_prior,whitenoise_prior =None):
    '''
    A function to define a linear model in pyro

    ------------Inputs--------------
    X: 2D torch tensor with shape (n_samples,n_features)
    y: 1D torch tensor with shape (n_samples)
    x_sigma: float, standard deviation of the error for age, which is obtained from the age data model
    y_sigma: float, standard deviation or covariance function of the error for the RSL, which is obtained from the RSL data model
    intercept_prior: pyro distribution for the intercept coefficient
    coefficient_prior: pyro distribution for the slope coefficient
    whitenoise_prior: pyro distribution for the white noise

    '''
    # Define our intercept prior
   
    linear_combination = pyro.sample("b", intercept_prior)
    #Define our coefficient prior
   
    beta_coef = pyro.sample("a", coefficient_prior)
    #generate random error for age
    N = X.shape[0]
    x_noise = pyro.sample('obs_xerr',dist.Normal(torch.zeros(N),x_sigma).to_event(1))
    x_noisy = X[:, 0]+x_noise
   
    #calculate mean prediction
    mean = linear_combination + (x_noisy * beta_coef)
    if whitenoise_prior == None:
        whitenoise = 0
    else:
        whitenoise = pyro.sample('whitenoise',whitenoise_prior)
    
    with pyro.plate("data", X.shape[0]):        
        # Condition the expected mean on the observed target y
        y_sigma = (y_sigma**2+whitenoise**2)**0.5
        if y_sigma.dim() <=1:
            pyro.sample("obs", dist.Normal(mean, y_sigma), obs=y)
        elif y_sigma.dim() ==2:
            pyro.sample("obs", dist.MultivariateNormal(mean, y_sigma), obs=y)





def linear_model_uniform(X, y,x_sigma,y_range,intercept_prior,coefficient_prior,whitenoise_prior):
    '''
    A function to define a linear model with uniform likelihood in pyro

    ------------Inputs--------------
    X: 2D torch tensor with shape (n_samples,n_features)
    y: 1D torch tensor with shape (n_samples), note y is the middle point of the paleo RSL range
    x_sigma: float, standard deviation of the error for age, which is obtained from the age data model
    y_range: float, upper and lower bound of the error for the paleo RSL, which is defined by different types of coral
    intercept_prior: pyro distribution for the intercept coefficient
    coefficient_prior: pyro distribution for the slope coefficient

    '''
    # Define our intercept prior
   
    linear_combination = pyro.sample("b", intercept_prior)
    #Define our coefficient prior
   
    beta_coef = pyro.sample("a", coefficient_prior)
    #generate random error for age
    N = X.shape[0]
    x_noise = pyro.sample('obs_xerr',dist.Normal(torch.zeros(N),x_sigma).to_event(1))
    x_noisy = X[:, 0]+x_noise
   
    #calculate mean prediction
    mean = linear_combination + (x_noisy * beta_coef)
    whitenoise = pyro.sample("whitenoise",whitenoise_prior)

    with pyro.plate("data", X.shape[0]):        
        # Condition the expected mean on the observed target y
        y_up = mean + y_range +whitenoise
        y_down = mean - y_range - whitenoise
        pyro.sample("obs", dist.Uniform(y_down,y_up), obs=y)

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
            if  noise.dim() ==1:
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
        Kff = Kff + self.noise
        Kff.view(-1)[:: N + 1] += self.jitter
#         Kff.view(-1)[:: N + 1] += self.jitter + self.noise  # add noise to diagonal

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
            prediction output or notest_cov[i:i+1]t.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")

        N = self.X.size(0)
        Kff = self.kernel(self.X).contiguous()
        if self.noise.dim() ==1:
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


    def iter_sample(self, noiseless=True):
        r"""
        Iteratively constructs a sample from the Gaussian Process posterior.

        Recall that at test input points :math:`X_{new}`, the posterior is
        multivariate Gaussian distributed with mean and covariance matrix
        given by :func:`forward`.

        This method samples lazily from this multivariate Gaussian. The advantage
        of this approach is that later query points can depend upon earlier ones.
        Particularly useful when the querying is to be done by an optimisation
        routine.

        .. note:: The noise parameter ``noise`` (:math:`\epsilon`) together with
            kernel's parameters have been learned from a training procedure (MCMC or
            SVI).

        :param bool noiseless: A flag to decide if we want to add sampling noise
            to the samples beyond the noise inherent in the GP posterior.
        :returns: sampler
        :rtype: function
        """
        noise = self.noise.detach()
        X = self.X.clone().detach()
        y = self.y.clone().detach()
        N = X.size(0)
        Kff = self.kernel(X).contiguous()
        Kff.view(-1)[:: N + 1] += noise  # add noise to the diagonal

        outside_vars = {"X": X, "y": y, "N": N, "Kff": Kff}

        def sample_next(xnew, outside_vars):
            """Repeatedly samples from the Gaussian process posterior,
            conditioning on previously sampled values.
            """
            warn_if_nan(xnew)

            # Variables from outer scope
            X, y, Kff = outside_vars["X"], outside_vars["y"], outside_vars["Kff"]

            # Compute Cholesky decomposition of kernel matrix
            Lff = torch.linalg.cholesky(Kff)
            y_residual = y - self.mean_function(X)

            # Compute conditional mean and variance
            loc, cov = conditional(
                xnew, X, self.kernel, y_residual, None, Lff, False, jitter=self.jitter
            )
            if not noiseless:
                cov = cov + noise

            ynew = torchdist.Normal(
                loc + self.mean_function(xnew), cov.sqrt()
            ).rsample()

            # Update kernel matrix
            N = outside_vars["N"]
            Kffnew = Kff.new_empty(N + 1, N + 1)
            Kffnew[:N, :N] = Kff
            cross = self.kernel(X, xnew).squeeze()
            end = self.kernel(xnew, xnew).squeeze()
            Kffnew[N, :N] = cross
            Kffnew[:N, N] = cross
            # No noise, just jitter for numerical stability
            Kffnew[N, N] = end + self.jitter
            # Heuristic to avoid adding degenerate points
            if Kffnew.logdet() > -15.0:
                outside_vars["Kff"] = Kffnew
                outside_vars["N"] += 1
                outside_vars["X"] = torch.cat((X, xnew))
                outside_vars["y"] = torch.cat((y, ynew))

            return ynew

        return lambda xnew: sample_next(xnew, outside_vars)


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
    :param torch.Tensor noise: Variance of Gaussian noise of this model.
    :param callable mean_function: An optional mean function :math:`m` of this Gaussian
        process. By default, we use zero mean.
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    """

    def __init__(self, X, y, xerr,kernel, noise=None, mean_function=None, jitter=1e-6):
        assert isinstance(
            X, torch.Tensor
        ), "X needs to be a torch Tensor instead of a {}".format(type(X))
        if y is not None:
            assert isinstance(
                y, torch.Tensor
            ), "y needs to be a torch Tensor instead of a {}".format(type(y))
       
        super().__init__(X, y,kernel, mean_function, jitter)
       
        if len(torch.tensor(xerr).shape)==0:
            xerr = torch.ones(len(X))*xerr
        self.xerr = xerr.double()
        self = self.double() #GP in pyro should use double precision
        self.X = self.X.double()
        self.y = self.y.double()

        if noise is None:
            self.noise = PyroParam(noise, constraints.positive)
        else:
            self.noise = noise.double()
    @pyro_method
    def model(self):
        self.set_mode("model")
        N = self.X.size(0)
        x_noise = pyro.sample('xerr',dist.Normal(torch.zeros(N),self.xerr**0.5).to_event(1))
        if self.X.dim()<=1:
            X_noisy = (self.X+x_noise)
        else:
            X_noisy = self.X
            X_noisy[:,0] += x_noise
       
        Kff = self.kernel(X_noisy)
        Kff.view(-1)[:: N + 1] += self.jitter + self.noise  # add noise to diagonal
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
        Kff.view(-1)[:: N + 1] += self.jitter + self.noise  # add noise to the diagonal
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

def change_point_model(X, y,x_sigma,y_sigma,n_cp,intercept_prior,coefficient_prior):
    '''
    A function to define a change-point model in pyro

    ------------Inputs--------------
    X: 2D torch tensor with shape (n_samples,n_features)
    y: 1D torch tensor with shape (n_samples)
    x_sigma: float, standard deviation of the error for age, which is obtained from the age data model
    y_sigma: float, standard deviation of the error for the RSL, which is obtained from the RSL datamodel
    n_cp: int, number of change-points
    intercept_prior: pyro distribution for the intercept coefficient
    coefficient_prior: pyro distribution for the slope coefficient

    '''
    # Define our intercept prior
    # intercept_prior = dist.Uniform(-5., 5.)
    b = pyro.sample("b", intercept_prior)
    beta_coef_list = torch.zeros(n_cp+1)
    cp_loc_list = torch.zeros(n_cp)
    #Define our coefficient prior
    cp_loc_prior = torch.linspace(X[:,0].min(),X[:,0].max(),n_cp+1)
    gap = cp_loc_prior[1]-cp_loc_prior[0]
    for i in range(n_cp+1):
        # coefficient_prior = dist.Uniform(-0.01, 0.01)
        beta_coef = pyro.sample(f"a_{i}", coefficient_prior)    
        beta_coef_list[i] = beta_coef
        if i<n_cp:
            cp_prior = dist.Uniform(cp_loc_prior[i]-gap,cp_loc_prior[i+1]+gap)
            cp_loc = pyro.sample(f"cp_{i}", cp_prior)
            cp_loc_list[i] = cp_loc
    # cp_loc_list,cp_sort_index = cp_loc_list.sort()
    # beta_coef_list = beta_coef_list[cp_sort_index]

    #generate random error for age
    x_noise = torch.normal(0, x_sigma)
    x_noisy = X[:, 0]+x_noise

    mean = torch.zeros(X.shape[0])
    last_intercept = b
   
    for i in range(n_cp+1):
        if i==0:
            start_age = X[:,0].min()
            start_idx = 0
            end_age = cp_loc_list[i]
            end_idx = torch.where(x_noisy<end_age)[0][-1]+1
            last_change_point = start_age
        elif i==n_cp:
            start_age = cp_loc_list[i-1]
            start_idx = torch.where(x_noisy>=start_age)[0][0]
            end_age = X[:,0].max()
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
        observation = pyro.sample("obs", dist.Normal(mean, y_sigma), obs=y)

def get_change_point_posterior(guide,sample_number):
    num_cp = int(list(guide().keys())[-1][list(guide().keys())[-1].index('_')+1:])
    output_dict = dict()
    output_dict['b'] = np.zeros(sample_number)
    output_dict['a'] = np.zeros([sample_number,num_cp+1])
    output_dict['cp'] = np.zeros([sample_number,num_cp])
    test_cp = []
    for i in range(num_cp):
        test_cp.append(guide.median()['cp_'+str(i)].detach().numpy())
    cp_index = np.argsort(test_cp)
   
    for i in range(sample_number):
        store_beta = []
        store_cp = []
        posterior_samples = guide()
        for i2 in range(num_cp+1):
            store_beta.append(posterior_samples['a_'+str(i2)].detach().numpy())
            if i2 < num_cp:
                store_cp.append(posterior_samples['cp_'+str(i2)].detach().numpy())
        output_dict['b'][i] = posterior_samples['b'].detach().numpy()
        output_dict['a'][i] = np.array(store_beta)
        output_dict['cp'][i] = np.array(store_cp)[cp_index]
    return output_dict

def change_point_forward(n_cp,cp_loc_list,new_X,data_X,beta_coef_list,b):
    '''
    A function to calculate the forward model of the change-point model

    ------------Inputs--------------
    n_cp: int, number of change-points
    cp_loc_list: 1D torch tensor with shape (n_cp), the location of the change-points
    new_X: 2D torch tensor with shape (n_samples,n_features) for new data prediction
    data_X: 2D torch tensor with shape (n_samples,n_features) for training data
    beta_coef_list: 1D torch tensor with shape (n_cp+1), the slope coefficients
    b: float, the intercept coefficient
    '''
    last_intercept = b
    mean = torch.zeros(new_X.shape[0])
    for i in range(n_cp+1):
        if i==0:
            start_age = data_X[:,0].min()
            start_idx = 0
            end_age = cp_loc_list[i]
            end_idx = torch.where(new_X<end_age)[0][-1]+1
            last_change_point = start_age
        elif i==n_cp:
            start_age = cp_loc_list[i-1]
            start_idx = torch.where(new_X>=start_age)[0][0]
            end_age = new_X[:,0].max()
            end_idx = new_X.shape[0]
        else:
            start_age = cp_loc_list[i-1]
            start_idx = torch.where(new_X>=start_age)[0][0]
            end_age = cp_loc_list[i]
            end_idx = torch.where(new_X<end_age)[0][-1]+1
       
        mean[start_idx:end_idx] = beta_coef_list[i] * (new_X[start_idx:end_idx:,0]-last_change_point) + last_intercept
        last_intercept = beta_coef_list[i] * (end_age-last_change_point) + last_intercept
        last_change_point = end_age
    return mean

def ensemble_GIA_model(X, y,x_sigma,y_sigma,model_ensemble,model_age):
    '''
    A function to define a linear model in pyro

    ------------Inputs--------------
    X: 2D torch tensor with shape (n_samples,n_features)
    y: 1D torch tensor with shape (n_samples)
    x_sigma: float, standard deviation of the error for age, which is obtained from the age data model
    y_sigma: float, standard deviation of the error for the RSL, which is obtained from the RSL datamodel

    '''
    # Define our intercept prior
    # weight_facor_list = torch.zeros(model_ensemble.shape[0])
    # for i in range(model_ensemble.shape[0]):
    weight_facor_list = pyro.sample("W",dist.Dirichlet(torch.ones(model_ensemble.shape[0])))
    # weight_facor_list[i] = weight_factor
    #generate random error for age
    x_noise = torch.normal(0, x_sigma)
    x_noisy = X[:, 0]+x_noise
    #interpolate GIA model
    mean = torch.zeros(len(x_noisy))
    for i in range(model_ensemble.shape[0]):
        GIA_model = interpolate.interp1d(model_age,model_ensemble[i])
        x_noisy[x_noisy>X[:, 0].max()] = X[:, 0].max()
        x_noisy[x_noisy<X[:, 0].min()] = X[:, 0].min()

        #calculate mean prediction
       
        mean += torch.tensor(GIA_model(x_noisy)) *weight_facor_list[i]
   
    with pyro.plate("data", y.shape[0]):        
        # Condition the expected mean on the observed target y
        observation = pyro.sample("obs", dist.Normal(mean, y_sigma), obs=y)

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
   

#THis model can be implemented within a class
def linear_model(X, y,x_sigma,y_sigma,intercept_prior,coefficient_prior):
    '''
    A function to define a linear model in pyro

    ------------Inputs--------------
    X: 2D torch tensor with shape (n_samples,n_features)
    y: 1D torch tensor with shape (n_samples)
    x_sigma: float, standard deviation of the error for age, which is obtained from the age data model
    y_sigma: float, standard deviation of the error for the RSL, which is obtained from the RSL datamodel
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
    with pyro.plate("data", y.shape[0]):        
        # Condition the expected mean on the observed target y
        observation = pyro.sample("obs", dist.Normal(mean, y_sigma), obs=y)

#--------------------------------------3.2 Modelling Choice GP module-----------------------------------------------

def cal_geo_dist2(X,Z=None):
    '''
    A function to calculate the squared distance matrix between each pair of X.
    The function takes a PyTorch tensor of X and returns a matrix
    where matrix[i, j] represents the spatial distance between the i-th and j-th X.
   
    -------Inputs-------
    X: PyTorch tensor of shape (n, 2), representing n pairs of (lat, lon) X
    R: approximate radius of earth in km
   
    -------Outputs-------
    distance_matrix: PyTorch tensor of shape (n, n), representing the distance matrix
    '''
    if Z is None:
        Z = X

    # Convert coordinates to radians
    X = torch.tensor(X)
    Z = torch.tensor(Z)
    X_coordinates_rad = torch.deg2rad(X)
    Z_coordinates_rad = torch.deg2rad(Z)
   
    # Extract latitude and longitude tensors
    X_latitudes_rad = X_coordinates_rad[:, 0]
    X_longitudes_rad = X_coordinates_rad[:, 1]

    Z_latitudes_rad = Z_coordinates_rad[:, 0]
    Z_longitudes_rad = Z_coordinates_rad[:, 1]

        # Calculate differences in latitude and longitude
    dlat = X_latitudes_rad[:, None] - Z_latitudes_rad[None, :]
    dlon = X_longitudes_rad[:, None] - Z_longitudes_rad[None, :]
    # Apply Haversine formula
    a = torch.sin(dlat / 2) ** 2 + torch.cos(X_latitudes_rad[:, None]) * torch.cos(Z_latitudes_rad[None, :]) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return c**2
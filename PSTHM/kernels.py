#----------------------Define Functions---------------------------

import torch
from torch.distributions import constraints
from pyro.nn.module import PyroParam
import torch
from torch.distributions import constraints
from pyro.contrib.gp.kernels.kernel import Kernel
import numpy as np
import numbers

#some of the kernels here are based on pyro gp kenrels: https://docs.pyro.ai/en/stable/contrib.gp.html

#-------------------------Define kernel operations-------------------------



class Combination(Kernel):
    """
    Base class for kernels derived from a combination of kernels.

    :param Kernel kern0: First kernel to combine.
    :param kern1: Second kernel to combine.
    :type kern1: Kernel or numbers.Number
    """

    def __init__(self, kern0, kern1):
        if not isinstance(kern0, Kernel):
            raise TypeError(
                "The first component of a combined kernel must be a " "Kernel instance."
            )
        if not (isinstance(kern1, Kernel) or isinstance(kern1, numbers.Number)):
            raise TypeError(
                "The second component of a combined kernel must be a "
                "Kernel instance or a number."
            )

        active_dims = set(kern0.active_dims)
        if isinstance(kern1, Kernel):
            active_dims |= set(kern1.active_dims)
        active_dims = sorted(active_dims)
        input_dim = len(active_dims)
        super().__init__(input_dim, active_dims)

        self.kern0 = kern0
        self.kern1 = kern1



class Sum(Combination):
    """
    Returns a new kernel which acts like a sum/direct sum of two kernels.
    The second kernel can be a constant.
    """

    def forward(self, X, Z=None, diag=False):
        if isinstance(self.kern1, Kernel):
            return self.kern0(X, Z, diag=diag) + self.kern1(X, Z, diag=diag)
        else:  # constant
            return self.kern0(X, Z, diag=diag) + self.kern1



class Product(Combination):
    """
    Returns a new kernel which acts like a product/tensor product of two kernels.
    The second kernel can be a constant.
    """

    def forward(self, X, Z=None, diag=False):
        if isinstance(self.kern1, Kernel):
            return self.kern0(X, Z, diag=diag) * self.kern1(X, Z, diag=diag)
        else:  # constant
            return self.kern0(X, Z, diag=diag) * self.kern1



#-------------------------Define Spatio-temporal GP kernels-------------------------
def _torch_sqrt(x, eps=1e-12):
    """
    A convenient function to avoid the NaN gradient issue of :func:`torch.sqrt`
    at 0.
    """
    # Ref: https://github.com/pytorch/pytorch/issues/2421
    return (x + eps).sqrt()

def check_geo_dim(X):
    '''
    A function to check if the dimension of X is correct.

    -------Inputs-------
    X: PyTorch tensor input for Gaussian Process
    '''

    if (X.dim()!=2) or (X.shape[1]!=3):
        raise ValueError("The dimension of input X is not correct. If you use a spatio kernel, X should be in shape of n x 3: [age, lat, lon].")

def check_pseudo(X,Z):
    '''
    A function to make sure there's no correlation between pseudo data and real data.
    '''
    check_geo_dim(X)
    dis_fun = torch.outer(torch.tensor(X[:,1]).abs()<361,torch.tensor(Z[:,1]).abs()<361).double()
    return dis_fun

class Isotropy(Kernel):
    """
    Base class for a family of isotropic covariance kernels which are functions of the
    distance :math:`|x-z|/l`, where :math:`l` is the length-scale parameter.

    By default, the parameter ``lengthscale`` has size 1. To use the isotropic version
    (different lengthscale for each dimension), make sure that ``lengthscale`` has size
    equal to ``input_dim``.

    :param torch.Tensor lengthscale: Length-scale parameter of this kernel.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, s_lengthscale=None,active_dims=None,geo=False,sp=False):
        super().__init__(input_dim, active_dims)

        variance = torch.tensor(1.0) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)
        if geo==False:
            lengthscale = torch.tensor(1.0) if lengthscale is None else lengthscale
            self.lengthscale = PyroParam(lengthscale, constraints.positive)
        else:
            s_lengthscale = torch.tensor(1.0) if s_lengthscale is None else s_lengthscale
            self.s_lengthscale = PyroParam(s_lengthscale, constraints.positive)
        if sp == True:
            self.lengthscale = torch.tensor(1.0) 
            self.s_lengthscale = torch.tensor(1.0) 
            
        self.geo= geo
        self.sp = sp

    def _square_scaled_dist(self, X, Z=None):
        """
        Returns :math:`\|\frac{X-Z}{l}\|^2`.
        """
        if X.dim() >1: X = X[:,0]

        if Z is None:
            Z = X
        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        scaled_X = X / self.lengthscale
        scaled_Z = Z / self.lengthscale
        X2 = (scaled_X**2).sum(1, keepdim=True)
        Z2 = (scaled_Z**2).sum(1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.t())
        r2 = X2 - 2 * XZ + Z2.t()
        return r2.clamp(min=0)

    def _scaled_dist(self, X, Z=None):
        """
        Returns :math:`\|\frac{X-Z}{l}\|`.
        """
        return _torch_sqrt(self._square_scaled_dist(X, Z))

    def _diag(self, X):
        """
        Calculates the diagonal part of covariance matrix on active features.
        """
        return self.variance.expand(X.size(0))

    def _scaled_geo_dist2(self,X,Z=None):
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
        if Z is None: Z = X
        if X.dim()==2: X = X[:,1:] #use lat and lon to calculate spatial distance
        if Z.dim()==2: Z = Z[:,1:] #use lat and lon to calculate spatial distance

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

        # Calculate the distance matrix
        distance_matrix = c / self.s_lengthscale

        return distance_matrix**2
    
    def _scaled_geo_dist(self, X, Z=None):
        """
        Returns :geo distance between X
        """
        return _torch_sqrt(self._scaled_geo_dist2(X, Z))

class RBF(Isotropy):
    r"""
    Implementation of Radial Basis Function kernel:

        :math:`k(x,z) = \sigma^2\exp\left(-0.5 \times \frac{|x-z|^2}{l^2}\right).`

    .. note:: This kernel also has name `Squared Exponential` in literature.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, s_lengthscale=None,active_dims=None,geo=False):
        super().__init__(input_dim,variance, lengthscale,s_lengthscale, active_dims,geo)

    def forward(self, X, Z=None, diag=False):
        
        if diag:
            return self._diag(X)
        if Z is None: Z=X

        if self.geo==False:
            r2 = self._square_scaled_dist(X, Z)
            return self.variance * torch.exp(-0.5 * r2)
        else:
            check_geo_dim(X)
            r2 = self._scaled_geo_dist2(X,Z)
            #no correlation for points with latitude larger than 360, which suppose to be psuedo data
            dis_fun = check_pseudo(X,Z)

            return torch.exp(-0.5 * r2)*dis_fun
        
class RationalQuadratic(Isotropy):
    r"""
    Implementation of RationalQuadratic kernel:

        :math:`k(x, z) = \sigma^2 \left(1 + 0.5 \times \frac{|x-z|^2}{\alpha l^2}
        \right)^{-\alpha}.`

    :param torch.Tensor scale_mixture: Scale mixture (:math:`\alpha`) parameter of this
        kernel. Should have size 1.
    """

    def __init__(
        self,
        input_dim,
        variance=None,
        lengthscale=None,
        s_lengthscale=None,
        scale_mixture=None,
        active_dims=None,
        geo=False
    ):
        super().__init__(input_dim, variance, lengthscale,s_lengthscale, active_dims,geo)

        if scale_mixture is None:
            scale_mixture = torch.tensor(1.0)
        self.scale_mixture = PyroParam(scale_mixture, constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)
        
        if Z is None: Z=X

        if self.geo==False:
            r2 = self._square_scaled_dist(X, Z)
            return self.variance * (1 + (0.5 / self.scale_mixture) * r2).pow(
            -self.scale_mixture
        )
        else:
            check_geo_dim(X)
            r2 = self._scaled_geo_dist2(X,Z)
            return (1 + (0.5 / self.scale_mixture) * r2).pow(
            -self.scale_mixture
            )           
        



class Exponential(Isotropy):
    r"""
    Implementation of Exponential kernel:

        :math:`k(x, z) = \sigma^2\exp\left(-\frac{|x-z|}{l}\right).`
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, s_lengthscale=None,active_dims=None,geo=False):
        super().__init__(input_dim, variance, lengthscale, s_lengthscale, active_dims,geo)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)
        
        if Z is None: Z=X

        if self.geo==False:
            r = self._scaled_dist(X, Z)
            return self.variance * torch.exp(-r)
        else:
            check_geo_dim(X)
            r = self._scaled_geo_dist(X,Z)
            #no correlation for points with longtitude larger than 360, which suppose to be psuedo data
            dis_fun = check_pseudo(X,Z)

            return torch.exp(-r) * dis_fun
        

class Matern21(Isotropy):
    r"""
    Implementation of Matern21 kernel:

        :math:`k(x, z) = \sigma^2\exp\left(- \frac{|x-z|}{l}\right).`
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, s_lengthscale=None, active_dims=None,geo=False):
        super().__init__(input_dim, variance, lengthscale, s_lengthscale, active_dims,geo)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)
        
        if Z is None: Z=X

        if self.geo==False:
            r = self._scaled_dist(X, Z)
            return self.variance * torch.exp(-r)
        
        else:
            check_geo_dim(X)
            r = self._scaled_geo_dist(X,Z)
            #no correlation for points with longtitude larger than 360, which suppose to be psuedo data
            dis_fun = check_pseudo(X,Z)
            
            return torch.exp(-r)*dis_fun
        

class Matern32(Isotropy):
    r"""
    Implementation of Matern32 kernel:

        :math:`k(x, z) = \sigma^2\left(1 + \sqrt{3} \times \frac{|x-z|}{l}\right)
        \exp\left(-\sqrt{3} \times \frac{|x-z|}{l}\right).`
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, s_lengthscale=None, active_dims=None,geo=False):
        super().__init__(input_dim, variance, lengthscale, s_lengthscale, active_dims,geo)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)
        
        if Z is None: Z=X

        if self.geo==False:
            r = self._scaled_dist(X, Z)
            sqrt3_r = 3**0.5 * r
            return self.variance * (1 + sqrt3_r) * torch.exp(-sqrt3_r)
        else:
            check_geo_dim(X)
            r = self._scaled_geo_dist(X,Z)
            sqrt3_r = 3**0.5 * r
            #no correlation for points with longtitude larger than 360, which suppose to be psuedo data
            dis_fun = check_pseudo(X,Z)

            return (1 + sqrt3_r) * torch.exp(-sqrt3_r) * dis_fun
        



class Matern52(Isotropy):
    r"""
    Implementation of Matern52 kernel:

        :math:`k(x,z)=\sigma^2\left(1+\sqrt{5}\times\frac{|x-z|}{l}+\frac{5}{3}\times
        \frac{|x-z|^2}{l^2}\right)\exp\left(-\sqrt{5} \times \frac{|x-z|}{l}\right).`
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, s_lengthscale=None, active_dims=None,geo=False):
        super().__init__(input_dim, variance, lengthscale, s_lengthscale, active_dims,geo)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)
        
        if Z is None: Z=X

        if self.geo==False:
            r2 = self._square_scaled_dist(X, Z)
            r = _torch_sqrt(r2)
            sqrt5_r = 5**0.5 * r
            return self.variance * (1 + sqrt5_r + (5 / 3) * r2) * torch.exp(-sqrt5_r)
        else:
            check_geo_dim(X)
            r2 = self._scaled_geo_dist2(X,Z)
            r = _torch_sqrt(r2)
            sqrt5_r = 5**0.5 * r
            #no correlation for points with longtitude larger than 360, which suppose to be psuedo data
            dis_fun = check_pseudo(X,Z)

            return (1 + sqrt5_r + (5 / 3) * r2) * torch.exp(-sqrt5_r) * dis_fun


class WhiteNoise(Kernel):
    r"""
    Implementation of WhiteNoise kernel:

        :math:`k(x, z) = \sigma^2 \delta(x, z),`

    where :math:`\delta` is a Dirac delta function.
    """

    def __init__(self, input_dim, variance=None, active_dims=None):
        super().__init__(input_dim, active_dims)

        variance = torch.tensor(1.0) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self.variance.expand(X.size(0))

        if Z is None:
            return self.variance.expand(X.size(0)).diag()
        else:
            return X.data.new_zeros(X.size(0), Z.size(0))
    
class WhiteNoise_SP(Isotropy):
    r"""
    Implementation of WhiteNoise kernel with multiple choices of spatial and temporal correlation:

    if geo==True, then the kernel is spatially uncorrelated
    if geo==False, then the kernel is a whiet noise kernel

    
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, s_lengthscale=None, active_dims=None,geo=False,sp=False):
        super().__init__(input_dim, variance, lengthscale, s_lengthscale, active_dims,geo,sp)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)
        
        if Z is None: Z=X
        
        # if self.sp==True:
        #     tem_delta_fun = self._scaled_dist(X, Z)<1e-4
        #     sp_delta_fun = (self._scaled_geo_dist(X,Z)<1e-4) & torch.outer(X[:,2]<361,Z[:,2]<361)

        #     return self.variance * (tem_delta_fun * sp_delta_fun).double()
        
        if self.geo==True:
            delta_fun = self._scaled_geo_dist(X,Z)<1e-4 
            #no correlation for points with longtitude larger than 360, which suppose to be psuedo data
            dis_fun = torch.outer(torch.tensor(X)[:,1].abs()<361,torch.tensor(Z)[:,1].abs()<361).double()

            return self.variance * delta_fun.double() * dis_fun
        
        if self.geo==False:
            delta_fun = self._scaled_dist(X, Z)<1e-4  & torch.outer(X[:,2]<361,Z[:,2]<361)
            return self.variance * delta_fun.double()


class Constant(Kernel):
    r"""
    Implementation of Constant kernel:

        :math:`k(x, z) = \sigma^2.`
    """

    def __init__(self, input_dim, variance=None, active_dims=None):
        super().__init__(input_dim, active_dims)

        variance = torch.tensor(1.0) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self.variance.expand(X.size(0))

        if Z is None:
            Z = X
        return self.variance.expand(X.size(0), Z.size(0))


class Cosine(Isotropy):
    r"""
    Implementation of Cosine kernel:

        :math:`k(x,z) = \sigma^2 \cos\left(\frac{|x-z|}{l}\right).`

    :param torch.Tensor lengthscale: Length-scale parameter of this kernel.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None):
        super().__init__(input_dim, variance, lengthscale, active_dims)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)

        r = self._scaled_dist(X, Z)
        return self.variance * torch.cos(r)



class Periodic(Kernel):
    r"""
    Implementation of Periodic kernel:

        :math:`k(x,z)=\sigma^2\exp\left(-2\times\frac{\sin^2(\pi(x-z)/p)}{l^2}\right),`

    where :math:`p` is the ``period`` parameter.

    References:

    [1] `Introduction to Gaussian processes`,
    David J.C. MacKay

    :param torch.Tensor lengthscale: Length scale parameter of this kernel.
    :param torch.Tensor period: Period parameter of this kernel.
    """

    def __init__(
        self, input_dim, variance=None, lengthscale=None, period=None, active_dims=None
    ):
        super().__init__(input_dim, active_dims)

        variance = torch.tensor(1.0) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)

        lengthscale = torch.tensor(1.0) if lengthscale is None else lengthscale
        self.lengthscale = PyroParam(lengthscale, constraints.positive)

        period = torch.tensor(1.0) if period is None else period
        self.period = PyroParam(period, constraints.positive)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self.variance.expand(X.size(0))

        if Z is None:
            Z = X
        X = self._slice_input(X)
        Z = self._slice_input(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        d = X.unsqueeze(1) - Z.unsqueeze(0)
        scaled_sin = torch.sin(np.pi * d / self.period) / self.lengthscale
        return self.variance * torch.exp(-2 * (scaled_sin**2).sum(-1))


class DotProduct(Kernel):
    r"""
    Base class for kernels which are functions of :math:`x \cdot z`.
    """

    def __init__(self, input_dim,ref_year=None, variance=None, active_dims=None):
        super().__init__(input_dim, active_dims)

        variance = torch.tensor(1.0) if variance is None else variance
        self.variance = PyroParam(variance, constraints.positive)
        ref_year = torch.tensor(0.) if ref_year is None else ref_year
        self.ref_year = ref_year
    def _dot_product(self, X, Z=None, diag=False):
        r"""
        Returns :math:`X \cdot Z`.
        """
        if Z is None:
            Z = X
        X = self._slice_input(X)
        if diag:
            return (X**2).sum(-1)

        Z = self._slice_input(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        return X.matmul(Z.t())



class Linear(DotProduct):
    r"""
    Implementation of Linear kernel:

        :math:`k(x, z) = \sigma^2 x \cdot z.`

    Doing Gaussian Process regression with linear kernel is equivalent to doing a
    linear regression.

    .. note:: Here we implement the homogeneous version. To use the inhomogeneous
        version, consider using :class:`Polynomial` kernel with ``degree=1`` or making
        a :class:`.Sum` with a :class:`.Constant` kernel.
    """

    def __init__(self, input_dim,ref_year = None, variance=None, active_dims=None):
        super().__init__(input_dim,ref_year, variance, active_dims)

    def forward(self, X, Z=None, diag=False):
        if X.dim() >1: X = X[:,0]
        if Z ==None: Z = X
        if Z.dim() >1: Z = Z[:,0]
        return self.variance * self._dot_product(X-self.ref_year, Z-self.ref_year, diag)


class Polynomial(DotProduct):
    r"""
    Implementation of Polynomial kernel:

        :math:`k(x, z) = \sigma^2(\text{bias} + x \cdot z)^d.`

    :param torch.Tensor bias: Bias parameter of this kernel. Should be positive.
    :param int degree: Degree :math:`d` of the polynomial.
    """

    def __init__(self, input_dim, variance=None, bias=None, degree=1, active_dims=None):
        super().__init__(input_dim, variance, active_dims)

        bias = torch.tensor(1.0) if bias is None else bias
        self.bias = PyroParam(bias, constraints.positive)
        self.variance=variance
        if not isinstance(degree, int) or degree < 1:
            raise ValueError(
                "Degree for Polynomial kernel should be a positive integer."
            )
        self.degree = degree

    def forward(self, X, Z=None, diag=False):
        if X.dim() >1: X = X[:,0]
        if Z ==None: Z = X
        if Z.dim() >1: Z= Z[:,0] 
        return self.variance * (
            (self.bias + self._dot_product(X, Z, diag)) ** self.degree
        )


class Empirical(Kernel):
    r"""
    Implementation of an empirical kernel where the input is a physical model ensemble,
    and the kernel is the sampling covariance matrix of the ensemble. It includes a temporal gaussian tamper function, 
    to use it, user need to define taper length.

    :param int input_dim: Dimension of input vectors.
    :param list model_ensemble: A list of callable models that make up the ensemble.
    :param np.ndarray weights: Optional weights for each model in the ensemble.
    :param list active_dims: Indices of active dimensions.
    :param int tau: temporal value for temporl gaussian taper function
    """
    def __init__(self, input_dim, model_ensemble=None, weights=None, active_dims=None, tau=None):
        super().__init__(input_dim, active_dims)
        self.ensemble = model_ensemble
        self.weights = np.ones(len(model_ensemble)) if weights is None else weights
        self.tau = tau

    def forward(self, X, Z=None, diag=False):
        # Collect predictions from each model in the ensemble
        all_pred = torch.stack([torch.tensor(model(X)) for model in self.ensemble])

        if Z is None:
            Z = X

        # Handle tapering
        if self.tau is not None:
            # Apply Gaussian taper based on the distance
            if X.dim() == 1:
                distance_matrix = (X[:, None] - Z[None, :])**2
            elif X.dim()>1:
                distance_matrix = (X[:,0][:, None] - Z[:,0][None, :])**2
            
            taper_matrix = torch.exp(-distance_matrix / (self.tau**2))
        else:
            taper_matrix = 1
        if not diag:
            if X is Z:
                cov_matrix = np.cov(all_pred.detach().numpy(), aweights=self.weights, rowvar=False)
            else:
                all_pred_Z = torch.stack([torch.tensor(model(Z)) for model in self.ensemble])
                cov_matrix = np.cov(np.hstack([all_pred.detach().numpy(), all_pred_Z.detach().numpy()]),
                                    aweights=self.weights, rowvar=False)
                n = all_pred.shape[1]
                m = all_pred_Z.shape[1]
                cov_matrix = cov_matrix[:n, n:n+m]

            # Apply tapering
            cov_matrix = torch.tensor(cov_matrix) * taper_matrix
            return cov_matrix
        else:
            cov_matrix = np.cov(all_pred.detach().numpy(), aweights=self.weights, rowvar=False)
            cov_matrix = torch.tensor(cov_matrix).diag()

            # Apply tapering if needed
            return cov_matrix * taper_matrix.diag() if self.tau is not None else cov_matrix
#----------------------Define Functions---------------------------

import torch
from torch.distributions import constraints
from pyro.nn.module import PyroParam
import torch
from torch.distributions import constraints
from pyro.contrib.gp.kernels.kernel import Kernel

#-------------------------Define Spatio-temporal GP kernels-------------------------
def _torch_sqrt(x, eps=1e-12):
    """
    A convenient function to avoid the NaN gradient issue of :func:`torch.sqrt`
    at 0.
    """
    # Ref: https://github.com/pytorch/pytorch/issues/2421
    return (x + eps).sqrt()


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
            r2 = self._square_scaled_dist(X[:,:1], Z[:,:1])
            return self.variance * torch.exp(-0.5 * r2)
        else:
            r2 = self._scaled_geo_dist2(X[:,1:],Z[:,1:])
            #no correlation for points with longtitude larger than 360, which suppose to be psuedo data
            dis_fun = torch.outer(torch.tensor(X)[:,1].abs()<361,torch.tensor(Z)[:,1].abs()<361).double()

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
            r2 = self._square_scaled_dist(X[:,:1], Z[:,:1])
            return self.variance * (1 + (0.5 / self.scale_mixture) * r2).pow(
            -self.scale_mixture
        )
        else:
            r2 = self._scaled_geo_dist2(X[:,1:],Z[:,1:])
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
            r = self._scaled_dist(X[:,:1], Z[:,:1])
            return self.variance * torch.exp(-r)
        else:
            r = self._scaled_geo_dist(X[:,1:],Z[:,1:])
            #no correlation for points with longtitude larger than 360, which suppose to be psuedo data
            dis_fun = torch.outer(torch.tensor(X)[:,1].abs()<361,torch.tensor(Z)[:,1].abs()<361).double()

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
            r = self._scaled_dist(X[:,:1], Z[:,:1])
            return self.variance * torch.exp(-r)
        
        else:
            r = self._scaled_geo_dist(X[:,1:],Z[:,1:])
            #no correlation for points with longtitude larger than 360, which suppose to be psuedo data
            dis_fun = torch.outer(torch.tensor(X)[:,1].abs()<361,torch.tensor(Z)[:,1].abs()<361).double()
            
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
            r = self._scaled_dist(X[:,:1], Z[:,:1])
            sqrt3_r = 3**0.5 * r
            return self.variance * (1 + sqrt3_r) * torch.exp(-sqrt3_r)
        else:
            r = self._scaled_geo_dist(X[:,1:],Z[:,1:])
            sqrt3_r = 3**0.5 * r
            #no correlation for points with longtitude larger than 360, which suppose to be psuedo data
            dis_fun = torch.outer(torch.tensor(X)[:,1].abs()<361,torch.tensor(Z)[:,1].abs()<361).double()

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
            r2 = self._square_scaled_dist(X[:,:1], Z[:,:1])
            r = _torch_sqrt(r2)
            sqrt5_r = 5**0.5 * r
            return self.variance * (1 + sqrt5_r + (5 / 3) * r2) * torch.exp(-sqrt5_r)
        else:
            r2 = self._scaled_geo_dist2(X[:,1:],Z[:,1:])
            r = _torch_sqrt(r2)
            sqrt5_r = 5**0.5 * r
            #no correlation for points with longtitude larger than 360, which suppose to be psuedo data
            dis_fun = torch.outer(torch.tensor(X)[:,1].abs()<361,torch.tensor(Z)[:,1].abs()<361).double()

            return (1 + sqrt5_r + (5 / 3) * r2) * torch.exp(-sqrt5_r) * dis_fun
    
class WhiteNoise(Isotropy):
    r"""
    Implementation of WhiteNoise kernel:

        :math:`k(x, z) = \sigma^2 \delta(x, z),`

    where :math:`\delta` is a Dirac delta function.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, s_lengthscale=None, active_dims=None,geo=False,sp=False):
        super().__init__(input_dim, variance, lengthscale, s_lengthscale, active_dims,geo,sp)

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self._diag(X)
        
        if Z is None: Z=X
        
        if self.sp==True:
            tem_delta_fun = self._scaled_dist(X[:,:1], Z[:,:1])<1e-4
            sp_delta_fun = (self._scaled_geo_dist(X[:,1:],Z[:,1:])<1e-4) & torch.outer(X[:,2]<360,Z[:,2]<360)

            return self.variance * (tem_delta_fun * sp_delta_fun).double()
        
        if self.geo==True:
            delta_fun = self._scaled_geo_dist(X[:,1:],Z[:,1:])<1e-4
            #no correlation for points with longtitude larger than 360, which suppose to be psuedo data
            dis_fun = torch.outer(torch.tensor(X)[:,1].abs()<361,torch.tensor(Z)[:,1].abs()<361).double()

            return self.variance * delta_fun.double() * dis_fun
        
        if self.geo==False:
            delta_fun = self._scaled_dist(X[:,:1], Z[:,:1])<1e-4
            return self.variance * delta_fun.double()

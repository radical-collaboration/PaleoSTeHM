#----------------------Define Functions---------------------------

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from scipy import interpolate
import torch
from pyro.infer import Predictive

def gen_pred_matrix(age,lat,lon):
    '''
    A function to generate an input matrix for Spatio-temporal GP model

    ----------Inputs----------------
    age: a numpy array, age of the prediction points
    lat: a numpy array, latitude of the prediction points
    lon: a numpy array, longitude of the prediction points

    ----------Outputs----------------
    output_matrix: a torch tensor, input matrix for the spatio-temporal GP model
    '''
    age = np.array(age)
    lat = np.array(lat)
    lon = np.array(lon)

    lon_matrix,lat_matrix,age_matrix = np.meshgrid(lon,lat,age)
    
    output_matrix = torch.tensor(np.hstack([age_matrix.flatten()[:,None],lat_matrix.flatten()[:,None],lon_matrix.flatten()[:,None]])).double()
    return output_matrix

def cal_rate_var(test_X,cov_matrix,mean_rsl,difftimestep=200):
    '''A function to caluclate standard deviation of sea-levle change rate (i.e., first derivative of 
    GP).
    ------------------Inputs----------------------------
    test_X: an array of test input values
    cov_matrix: full covariance matrix from GP regression
    mean_rsl: GP regression produced mean RSL prediction
    difftimestep: time period for averaging 
    
    ------------------Outputs---------------------------
    difftimes: time series for the outputs
    rate: averaged sea-level change rate
    rate_sd: averaged sea-level change rate standard deviation
    '''
    
    Mdiff = np.array(np.equal.outer(test_X, test_X.T),dtype=int) - np.array(np.equal.outer(test_X, test_X.T + difftimestep),dtype=int)
    Mdiff = Mdiff * np.equal.outer(np.ones(len(test_X))*1, np.ones(len(test_X)))
    sub = np.where(np.sum(Mdiff, axis=1) == 0)[0]
    Mdiff = Mdiff[sub, :]
    difftimes = np.abs(Mdiff) @ test_X / np.sum(np.abs(Mdiff), axis=1)
    Mdiff = Mdiff / (Mdiff @ test_X.T)[:,None]
    rate_sd = np.sqrt(np.diag(Mdiff @ cov_matrix @ Mdiff.T))
    rate = Mdiff @ mean_rsl
    
    return difftimes,rate, rate_sd


def cal_spatiotemporal_rate_var(test_X,cov_matrix,mean_rsl,difftimestep=200):
    '''A function to caluclate standard deviation of sea-levle change rate (i.e., first derivative of 
    GP).
    ------------------Inputs----------------------------
    test_X: an array of test input values, either 1D (time) or 2D (time, lat, lon)
    cov_matrix: full covariance matrix from GP regression
    mean_rsl: GP regression produced mean RSL prediction
    difftimestep: time period for averaging 
    
    ------------------Outputs---------------------------
    difftimes: time series for the outputs
    rate: averaged sea-level change rate
    rate_var: averaged sea-level change rate covariance matrix
    rate_sd: averaged sea-level change rate standard deviation
    '''
    if len(test_X.shape) == 1:
        
        Mdiff = np.array(np.equal.outer(test_X, test_X.T),dtype=int) - np.array(np.equal.outer(test_X, test_X.T + difftimestep),dtype=int)
    else:
        Mdiff = np.array(np.equal.outer(test_X[:,0], test_X[:,0].T),dtype=int) - np.array(np.equal.outer(test_X[:,0], test_X[:,0].T + difftimestep),dtype=int)
        Mdiff_2 = np.array(np.equal.outer(test_X[:,1], test_X[:,1].T),dtype=int)
        Mdiff_3 = np.array(np.equal.outer(test_X[:,2], test_X[:,2].T),dtype=int)
        Mdiff = Mdiff * Mdiff_2 * Mdiff_3
    sub = np.where(np.sum(Mdiff, axis=1) == 0)[0]
    Mdiff = Mdiff[sub, :]
    if len(test_X.shape) == 1:
        difftimes = np.abs(Mdiff) @ test_X / np.sum(np.abs(Mdiff), axis=1)
        Mdiff = Mdiff / (Mdiff @ test_X.T)[:,None]
    else:
        difftimes = np.abs(Mdiff) @ test_X[:,0] / np.sum(np.abs(Mdiff), axis=1)
        Mdiff = Mdiff / (Mdiff @ test_X[:,0].T)[:,None]
    rate_var = Mdiff @ cov_matrix @ Mdiff.T
    rate_sd = np.sqrt(np.diag(rate_var))
    rate = Mdiff @ mean_rsl
    
    return difftimes,rate, rate_var,rate_sd


def decompose_kernels(gpr,pred_matrix,kernels):
    '''
    A function to calculate different kernels contribution to final prediction

    ------------------Inputs----------------------------
    gpr: an optimized pyro GP regression model
    pred_matrix: a torch tensor of prediction matrix containing the data points for prediction
    kernels: a list of pyro kernels for decomposition

    ------------------Outputs---------------------------
    output: a list of tuples, each tuple contains the mean and covariance of the prediction for each kernel 
    '''
    N = len(gpr.X)
    M = pred_matrix.size(0)
    f_loc = gpr.y - gpr.mean_function(gpr.X)
    latent_shape = f_loc.shape[:-1]
    loc_shape = latent_shape + (M,)
    f_loc = f_loc.permute(-1, *range(len(latent_shape)))
    f_loc_2D = f_loc.reshape(N, -1)
    loc_shape = latent_shape + (M,)
    v_2D = f_loc_2D
    Kff = gpr.kernel(gpr.X).contiguous()
    if gpr.noise.dim() <=1:
        Kff.view(-1)[:: N + 1] += gpr.jitter + gpr.noise   # add noise to the diagonal
    elif gpr.noise.dim() ==2:
        Kff = Kff + gpr.noise
        Kff.view(-1)[:: N + 1] += gpr.jitter 
    Lff = torch.linalg.cholesky(Kff)
    
    output = []
    for kernel in kernels:
        Kfs = kernel(gpr.X, pred_matrix)
        pack = torch.cat((f_loc_2D, Kfs), dim=1)
        Lffinv_pack = pack.triangular_solve(Lff, upper=False)[0]
        W = Lffinv_pack[:, f_loc_2D.size(1):f_loc_2D.size(1) + M].t()
        Qss = W.matmul(W.t())
        v_2D = Lffinv_pack[:, :f_loc_2D.size(1)]
        loc = W.matmul(v_2D).t().reshape(loc_shape)
        Kss = kernel(pred_matrix)
        cov = Kss - Qss
        output.append((loc, cov))
        
    return output

def cal_misfit(y,y_sigma,prediction):
    
    return np.mean(np.sqrt(((y-prediction)/y_sigma)**2))

def cal_likelihood(y,y_std,pred):
    '''A function used to calcualte log likelihood function for a given prediction.
    This calculation only considers uncertainty in y axis. 
    
    ------------Inputs------------------
    y: reconstructed rsl
    y_std: standard deviation of reconstructed rsl
    pred: mean predction of rsl
    
    ------------Outputs------------------
    likelihood: mean likelihood of prediction fit to observation
    '''
    from scipy.stats import norm

    log_likelihood = 1 
    for i in range(len(y)):
        
        norm_dis = norm(y[i], y_std[i])
        log_likelihood+=np.log(norm_dis.pdf(pred[i]))
    
    return log_likelihood

def cal_MSE(y,yhat):
    '''
    A function to calculate MSE coefficient
    '''
    MSE = np.sum((yhat-y)**2)/len(y)
    return MSE


def cal_wMSE(y,yhat,y_sigma):
    '''
    A function to calculate weighted MSE coefficient
    '''
    wMSE = np.sum((yhat-y)**2/y_sigma**2)/len(y)
    return wMSE
    

def get_change_point_posterior(guide, sample_number):
    '''
    Function to sample from the guide and return posterior samples for change point model
    
    ----------Inputs----------
    guide: pyro model guide
    sample_number: int, number of samples to draw from the guide
    
    '''
    # Initialize dictionary to store posterior samples
    output_dict = dict()
    num_cp = int(list(guide().keys())[-2][list(guide().keys())[-2].index('_')+1:])
    # Extract median values to determine the number of change points and setup output structure
    median_values = guide.median()
    a_keys = ['a_'+str(i) for i in range(num_cp+1)]
    cp_keys = ['cp_'+str(i) for i in range(num_cp)]
    num_cp = len(cp_keys)
    num_as = len(a_keys)

    # Initialize arrays for posterior samples
    output_dict['b'] = np.zeros(sample_number)
    output_dict['a'] = np.zeros((sample_number, num_as))
    output_dict['cp'] = np.zeros((sample_number, num_cp))
    
    # Get the sorted index for change points to maintain order
    cp_values = [median_values[cp_key].item() for cp_key in cp_keys]
    cp_index = np.argsort(cp_values)

    # Sample from the guide and store results
    for i in range(sample_number):
        posterior_samples = guide()
        output_dict['b'][i] = posterior_samples['b'].detach().numpy()
        output_dict['a'][i] = np.array([posterior_samples[a_keys[j]].detach().numpy() for j in range(num_as)])
        output_dict['cp'][i] = np.array([posterior_samples[cp_keys[j]].detach().numpy() for j in cp_index])
    
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
    cp_loc_prior = torch.linspace(data_X[:,0].min(),data_X[:,0].max(),n_cp+3)[1:-1]
    gap = cp_loc_prior[1]-cp_loc_prior[0]
    initial_age, ending_age = data_X[:,0].min()-gap,data_X[:,0].max()+gap
    mean = torch.zeros(new_X.shape[0])
    for i in range(n_cp+1):
        if i==0:
            start_age = initial_age
            start_idx = 0
            end_age = cp_loc_list[i]
            end_idx = torch.where(new_X<end_age)[0][-1]+1
            last_change_point = start_age
        elif i==n_cp:
            start_age = cp_loc_list[i-1]
            start_idx = torch.where(new_X>=start_age)[0][0]
            end_age = ending_age
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

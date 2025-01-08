#----------------------Define Functions---------------------------

import torch
from torch.distributions import constraints
from pyro.nn.module import PyroParam
import torch
from torch.distributions import constraints
from pyro.contrib.gp.kernels.kernel import Kernel
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs
sys.path.append('../..')
import PSTHM 
import pyro.distributions as dist
import pyro 
import cartopy
import os

def load_rsl_data(csv_file_name):
    '''
    A function to load the RSL data from a csv file

    ---------------------   
    Input: 
        csv_file_name: str
            The name of the csv file containing the data
    ---------------------
    Output:
        rsl_age: np.array
            The age of the RSL data
        rsl: np.array
            The RSL data
        rsl_sigma: np.array
            The uncertainty in the RSL data
        rsl_age_sigma: np.array
            The uncertainty in the age of the RSL data
        ml_index: np.array
            The marine limiting index
        tl_index: np.array
            The terrestrial limiting index
        slip_index: np.array
            The slip index
        X_all: np.array
            The full data set
        rsl_region: np.array
            The regional number
        rsl_regional_name: np.array, obj
            The regional name
        y: np.array
            The RSL data

    '''
    #load the data
    rsl_data = pd.read_csv (csv_file_name)
    rsl_region = rsl_data['Region.1']
    rsl_region_name = rsl_data['Region']
    rsl_lat = rsl_data['Latitude']
    rsl_lon = rsl_data['Longitude']
    rsl_type = rsl_data['Limiting']
    rsl = rsl_data['RSL']
    rsl_p2sigma = rsl_data['RSLer_up_2sd']
    rsl_m2sigma = rsl_data['RSLer_low_2sd']
    rsl_1sigma_stack = np.hstack([rsl_m2sigma.values[:,None],rsl_p2sigma.values[:,None]])/2
    rsl_sigma = (rsl_p2sigma+rsl_m2sigma)/4

    rsl_age = rsl_data['Age']
    rsl_age_p2sigma = rsl_data['Age_up_er_2sd']
    rsl_age_m2sigma = rsl_data['Age_low_er_2sd']
    rsl_age_1sigma_stack = np.hstack([rsl_age_m2sigma.values[:,None],rsl_age_p2sigma.values[:,None]])/2
    rsl_age_sigma = (rsl_age_p2sigma+rsl_age_m2sigma)/4
    ml_index =  rsl_type==-1
    tl_index =  rsl_type==1
    slip_index = rsl_type==0
    X_all = np.hstack([rsl_age.values[:,None][slip_index],rsl_lat.values[:,None][slip_index],rsl_lon.values[:,None][slip_index]])
    y = rsl.values[slip_index]

    return [rsl_age, rsl, rsl_sigma,rsl_1sigma_stack, rsl_age_sigma,rsl_age_1sigma_stack, ml_index, tl_index, slip_index, X_all, rsl_region, rsl_region_name, y,rsl_lat,rsl_lon]


def implement_sp_gp_model(X_all, y, rsl_sigma, rsl_age_sigma,iteration=1000):
    '''
    A function to implement the spatio-temporal Gaussian Process model for sea-level reconstruction.
    The function takes the following inputs:
    X_all: A numpy array containing the age of the sea-level data points.
    y: A numpy array containing the sea-level data points.
    rsl_sigma: A numpy array containing the uncertainty in the sea-level data points.
    rsl_age_sigma: A numpy array containing the uncertainty in the age of the sea-level data points.
    iteration: An integer specifying the number of iterations for optimization.

    The function returns the optimized Gaussian Process model.

    '''
    pyro.clear_param_store()
    #define global temporal kernel
    global_kernel = PSTHM.kernels.Matern32 (input_dim=1,geo=False)
    global_kernel.set_prior("lengthscale", dist.Uniform(torch.tensor(100.), torch.tensor(20000.)))
    global_kernel.set_prior("variance", dist.Uniform(torch.tensor(1e-5), torch.tensor(1000.)))

    #define regionally linar spatio-temporal kernel
    regional_linear_temporal_kernel = PSTHM.kernels.Linear(input_dim=1)
    regional_linear_temporal_kernel.set_prior("variance", dist.Uniform(torch.tensor(1e-12), torch.tensor(1e-6)))
    regional_linear_temporal_kernel.ref_year = torch.tensor(0.)
    regional_linear_spatial_kernel = PSTHM.kernels.Matern21(input_dim=1,geo=True)
    regional_linear_spatial_kernel.set_prior("s_lengthscale", dist.Uniform(torch.tensor(0.01), torch.tensor(0.5)))
    regional_linear_kernel = PSTHM.kernels.Product(regional_linear_temporal_kernel, regional_linear_spatial_kernel)

    #define regionally non-linar spatio-temporal kernel
    regional_nl_temporal_kernel = PSTHM.kernels.Matern32(input_dim=1,lengthscale =global_kernel.lengthscale,geo=False)
    regional_nl_temporal_kernel.set_prior("variance", dist.Uniform(torch.tensor(1e-6), torch.tensor(100.)))
    regional_nl_temporal_kernel.set_prior("lengthscale", dist.Uniform(torch.tensor(100.), torch.tensor(20000.)))
    regional_nl_spatial_kernel = PSTHM.kernels.Matern21(input_dim=1,geo=True)
    regional_nl_spatial_kernel.set_prior("s_lengthscale", dist.Uniform(torch.tensor(0.01), torch.tensor(0.5)))
    regional_nl_kernel = PSTHM.kernels.Product(regional_nl_temporal_kernel, regional_nl_spatial_kernel)

    #define regionally non-linar spatio-temporal kernel
    local_nl_temporal_kernel = PSTHM.kernels.Matern32(input_dim=1,lengthscale =global_kernel.lengthscale,geo=False)
    local_nl_temporal_kernel.set_prior("variance", dist.Uniform(torch.tensor(1e-6), torch.tensor(1.)))
    local_nl_temporal_kernel.set_prior("lengthscale", dist.Uniform(torch.tensor(100.), torch.tensor(20000.)))
    local_nl_spatial_kernel = PSTHM.kernels.Matern21(input_dim=1,geo=True)
    local_nl_spatial_kernel.set_prior("s_lengthscale", dist.Uniform(torch.tensor(0.001), torch.tensor(0.01)))
    local_nl_kernel = PSTHM.kernels.Product(local_nl_temporal_kernel, local_nl_spatial_kernel)


    #site specific datum correction
    sp_whitenoise_kernel = PSTHM.kernels.WhiteNoise_SP(input_dim=1,sp=False,geo=True)
    sp_whitenoise_kernel.set_prior('variance',dist.Uniform(torch.tensor(1e-7),torch.tensor(1.)))

    #whitenoise error
    whitenoise_kernel = PSTHM.kernels.WhiteNoise(input_dim=1)
    whitenoise_kernel.set_prior('variance',dist.Uniform(torch.tensor(1e-7),torch.tensor(1.)))

    combined_sp_kernel = PSTHM.kernels.Sum(global_kernel,regional_linear_kernel)
    combined_sp_kernel = PSTHM.kernels.Sum(combined_sp_kernel,regional_nl_kernel)
    combined_sp_kernel = PSTHM.kernels.Sum(combined_sp_kernel,local_nl_kernel)
    combined_sp_kernel = PSTHM.kernels.Sum(combined_sp_kernel,sp_whitenoise_kernel)
    combined_sp_kernel = PSTHM.kernels.Sum(combined_sp_kernel,whitenoise_kernel)

    gpr = PSTHM.model.GPRegression_V(torch.tensor(X_all), torch.tensor(y), combined_sp_kernel,noise=torch.tensor(rsl_sigma.values),jitter=1e-5)
    print('-----------------------------------')
    print('Model implementation complete successfully')

    gpr,track_list = PSTHM.opti.SVI_NI_optm(gpr,x_sigma=torch.tensor(rsl_age_sigma.values),num_iteration=iteration,lr=0.2)
        
    optimized_params = {}

    # Extract optimized parameters for global kernel
    optimized_params['global_kernel_lengthscale'] = global_kernel.lengthscale.item()
    optimized_params['global_kernel_variance'] = global_kernel.variance.item()

    # Extract optimized parameters for regional linear temporal kernel
    optimized_params['regional_linear_temporal_kernel_variance'] = regional_linear_temporal_kernel.variance.item()

    # Extract optimized parameters for regional linear spatial kernel
    optimized_params['regional_linear_spatial_kernel_s_lengthscale'] = regional_linear_spatial_kernel.s_lengthscale.item()

    # Extract optimized parameters for regional non-linear temporal kernel
    optimized_params['regional_nl_temporal_kernel_variance'] = regional_nl_temporal_kernel.variance.item()
    optimized_params['regional_nl_temporal_kernel_lengthscale'] = regional_nl_temporal_kernel.lengthscale.item()

    # Extract optimized parameters for regional non-linear spatial kernel
    optimized_params['regional_nl_spatial_kernel_s_lengthscale'] = regional_nl_spatial_kernel.s_lengthscale.item()

    # Extract optimized parameters for local non-linear temporal kernel
    optimized_params['local_nl_temporal_kernel_variance'] = local_nl_temporal_kernel.variance.item()
    optimized_params['local_nl_temporal_kernel_lengthscale'] = local_nl_temporal_kernel.lengthscale.item()

    # Extract optimized parameters for local non-linear spatial kernel
    optimized_params['local_nl_spatial_kernel_s_lengthscale'] = local_nl_spatial_kernel.s_lengthscale.item()

    # Extract optimized parameters for site-specific white noise kernel
    optimized_params['sp_whitenoise_kernel_variance'] = sp_whitenoise_kernel.variance.item()

    # Extract optimized parameters for white noise kernel
    optimized_params['whitenoise_kernel_variance'] = whitenoise_kernel.variance.item()

    # Create a DataFrame to display the results in a tabular format
    df_optimized_params = pd.DataFrame(list(optimized_params.items()), columns=['Kernel', 'Optimized Value'])

    optimized_params = {}

    # Extract optimized parameters for global kernel
    optimized_params['global_kernel_lengthscale (yr)'] = global_kernel.lengthscale.item()
    optimized_params['global_kernel_variance (m^2)'] = global_kernel.variance.item()

    # Extract optimized parameters for regional linear temporal kernel
    optimized_params['regional_linear_kernel_variance (m^2)'] = regional_linear_temporal_kernel.variance.item()

    # Extract optimized parameters for regional linear spatial kernel
    optimized_params['regional_linearl_kernel_spatil_lengthscale (km)'] = regional_linear_spatial_kernel.s_lengthscale.item()*6471

    # Extract optimized parameters for regional non-linear temporal kernel
    optimized_params['regional_nl_kernel_variance (m^2)'] = regional_nl_temporal_kernel.variance.item()
    optimized_params['regional_nl_kernel_lengthscale (yr)'] = regional_nl_temporal_kernel.lengthscale.item()

    # Extract optimized parameters for regional non-linear spatial kernel
    optimized_params['regional_nl_kernel_spatial_lengthscale (km)'] = regional_nl_spatial_kernel.s_lengthscale.item()*6471

    # Extract optimized parameters for local non-linear temporal kernel
    optimized_params['local_nl_kernel_variance (m^2)'] = local_nl_temporal_kernel.variance.item()
    optimized_params['local_nl_kernel_lengthscale (yr)'] = local_nl_temporal_kernel.lengthscale.item()

    # Extract optimized parameters for local non-linear spatial kernel
    optimized_params['local_nl_spatial_spatial_lengthscale (km)'] = local_nl_spatial_kernel.s_lengthscale.item()*6471

    # Extract optimized parameters for site-specific white noise kernel
    optimized_params['sp_whitenoise_kernel_variance (m^2)'] = sp_whitenoise_kernel.variance.item()

    # Extract optimized parameters for white noise kernel
    optimized_params['whitenoise_kernel_variance (m^2)'] = whitenoise_kernel.variance.item()

    # Create a DataFrame to display the results in a tabular format
    df_optimized_params = pd.DataFrame(list(optimized_params.items()), columns=['Kernel', 'Optimized Value'])
    print('Optimized parameters:',df_optimized_params)
    #check if the directory exists
    if not os.path.exists('Outputs/Hyperparameters/'):
        os.makedirs('Outputs/Hyperparameters/')
    df_optimized_params.to_csv('Outputs/Hyperparameters/optimized_params.csv', index=False)

    print('-----------------------------------')
    print('Optimization complete successfully')
    print('Optimized parameters have been saved to Outputs/Hyperparameters/optimized_params.csv')
    return gpr



def process_rsl_data(rsl_region, rsl_region_name, rsl_lat, rsl_lon, rsl_age, rsl, rsl_age_1sigma_stack, rsl_1sigma_stack, rsl_sigma, slip_index, tl_index, ml_index, gpr, test_age=None):
    """
    A function to process regional sea level (RSL) data, perform predictions, decompose kernels, 
    plot temporal data, and save results into a NetCDF file.

    Parameters:
    - rsl_region, rsl_region_name, rsl_lat, rsl_lon, rsl_age, rsl, rsl_age_1sigma_stack, rsl_1sigma_stack, rsl_sigma: Input RSL data arrays
    - slip_index, tl_index, ml_index: Index arrays for different data types
    - gpr: GP Regression model
    - test_age: Array of ages (optional)

    Outputs:
    - NetCDF file saved with processed results
    """
    if test_age is None:
        test_age = np.arange(12000, -200, -200)
        
    all_region,region_index = np.unique(rsl_region,return_index=True)
    all_region_name = rsl_region_name[region_index].values

    test_age = np.arange(12000,-200,-200)
    all_total_mean = np.zeros([len(all_region),len(test_age)])
    all_total_std = np.zeros([len(all_region),len(test_age)])
    all_global_mean = np.zeros([len(test_age)])
    all_global_std = np.zeros([len(test_age)])
    all_regional_nl_mean = np.zeros([len(all_region),len(test_age)])
    all_regional_nl_std = np.zeros([len(all_region),len(test_age)])
    all_regional_linear_mean = np.zeros([len(all_region),len(test_age)])
    all_regional_linear_std = np.zeros([len(all_region),len(test_age)])
    all_local_nl_mean = np.zeros([len(all_region),len(test_age)])
    all_local_nl_std = np.zeros([len(all_region),len(test_age)])

    all_total_rate_mean = np.zeros([len(all_region),len(test_age)-1])
    all_total_rate_std = np.zeros([len(all_region),len(test_age)-1])
    all_global_rate_mean = np.zeros([len(test_age)-1])
    all_global_rate_std = np.zeros([len(test_age)-1])
    all_regional_nl_rate_mean = np.zeros([len(all_region),len(test_age)-1])
    all_regional_nl_rate_std = np.zeros([len(all_region),len(test_age)-1])
    all_regional_linear_rate_mean = np.zeros([len(all_region),len(test_age)-1])
    all_regional_linear_rate_std = np.zeros([len(all_region),len(test_age)-1])
    all_local_nl_rate_mean = np.zeros([len(all_region),len(test_age)-1])
    all_local_nl_rate_std = np.zeros([len(all_region),len(test_age)-1])

    for i in range(len(all_region)):
        if slip_index[rsl_region==all_region[i]].sum()>0:
            ave_region_coord = np.mean((np.array([rsl_lat[rsl_region==all_region[i]].values,rsl_lon[rsl_region==all_region[i]].values]).T)[slip_index[rsl_region==all_region[i]]],axis=0)
        else:
            ave_region_coord = np.mean((np.array([rsl_lat[rsl_region==all_region[i]].values,rsl_lon[rsl_region==all_region[i]].values]).T),axis=0)
        pred_matrix = PSTHM.post.gen_pred_matrix(test_age,*ave_region_coord)
        y_mean, y_var = gpr(pred_matrix, full_cov=True)
        y_std = y_var.diag().sqrt()

        #-----------------define the kernel to decompose-----------------
        test_global_kernel =   gpr.kernel.kern0.kern0.kern0.kern0.kern0
        regional_linear_kernel =gpr.kernel.kern0.kern0.kern0.kern0.kern1
        regional_nl_kernel = gpr.kernel.kern0.kern0.kern0.kern1
        local_nl_kernel = gpr.kernel.kern0.kern0.kern1

        #-----------------decompose the kernel-----------------
        global_dep,reigonal_linear_dep,regional_nl_dep,local_nl_dep = PSTHM.post.decompose_kernels(gpr,pred_matrix,[test_global_kernel,regional_linear_kernel,regional_nl_kernel,local_nl_kernel])
        global_mean,global_var = global_dep
        global_std = global_var.diag().sqrt()

        regional_linear_mean,regional_linear_var = reigonal_linear_dep
        regional_linear_std = regional_linear_var.diag().sqrt()

        regional_nl_mean,regional_nl_var = regional_nl_dep
        regional_nl_std = regional_nl_var.diag().sqrt()

        local_nl_mean,local_nl_var = local_nl_dep
        local_nl_std = local_nl_var.diag().sqrt()

        plt.clf()
        fig=plt.figure(figsize=(50,20))
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        ax = plt.subplot(251)
        regional_slip_index = (rsl_region==all_region[i]) & slip_index
        regional_tl_index = (rsl_region==all_region[i]) & tl_index
        regional_ml_index = (rsl_region==all_region[i]) & ml_index
        PSTHM.plotting.plot_uncertainty_boxes(rsl_age[regional_slip_index],rsl[regional_slip_index], rsl_age_1sigma_stack[regional_slip_index]*2,rsl_1sigma_stack[regional_slip_index]*2,ax=ax)
        PSTHM.plotting.plot_limiting_data(rsl_age[regional_ml_index],rsl[regional_ml_index],rsl_age_1sigma_stack[regional_ml_index]*2,rsl_1sigma_stack[regional_ml_index]*2,marine_limiting=True,ax=ax)
        PSTHM.plotting.plot_limiting_data(rsl_age[regional_tl_index],rsl[regional_tl_index],rsl_age_1sigma_stack[regional_tl_index]*2,rsl_sigma[regional_tl_index]*2,marine_limiting=False,ax=ax)
        plt.plot(pred_matrix[:,0],y_mean.detach().numpy(),'C5',linewidth=3,label='Total Prediction')
        plt.fill_between(pred_matrix[:,0],y_mean.detach().numpy()-2*y_std.detach().numpy(),y_mean.detach().numpy()+2*y_std.detach().numpy(),color='C5',alpha=0.3,label='95% CI')
        plt.xlim(-100,12000)
        plt.xlabel('Age (BP)')
        plt.ylabel('RSL (m)')
        fig.legend(loc='upper left', bbox_to_anchor=(0.122, 0.95), bbox_transform=fig.transFigure, ncol=2, fontsize=20)
        plt.subplot(252)
        plt.plot(test_age,global_mean.detach().numpy(),'C0',linewidth=3,label='Global')
        plt.fill_between(test_age,global_mean.detach().numpy()-2*global_std.detach().numpy(),global_mean.detach().numpy()+2*global_std.detach().numpy(),color='C0',alpha=0.3,label='95% CI')
        plt.xlabel('Age (BP)')
        plt.ylabel('RSL (m)')
        plt.legend()
        plt.xlim(-100,12000)
        plt.subplot(253)
        plt.plot(test_age,regional_linear_mean.detach().numpy(),'C1',linewidth=3,label='Regional Linear')
        plt.fill_between(test_age,regional_linear_mean.detach().numpy()-2*regional_linear_std.detach().numpy(),regional_linear_mean.detach().numpy()+2*regional_linear_std.detach().numpy(),color='C1',alpha=0.3,label='95% CI')
        plt.xlabel('Age (BP)')
        plt.ylabel('RSL (m)')
        plt.legend()
        plt.xlim(-100,12000)
        plt.subplot(254)
        plt.plot(test_age,regional_nl_mean.detach().numpy(),'C2',linewidth=3,label='Regional Nonlinear')
        plt.fill_between(test_age,regional_nl_mean.detach().numpy()-2*regional_nl_std.detach().numpy(),regional_nl_mean.detach().numpy()+2*regional_nl_std.detach().numpy(),color='C2',alpha=0.3,label='95% CI')
        plt.xlabel('Age (BP)')
        plt.ylabel('RSL (m)')
        plt.legend()
        plt.xlim(-100,12000)
        plt.subplot(255)
        plt.plot(test_age,local_nl_mean.detach().numpy(),'C3',linewidth=3,label='Local Nonlinear')
        plt.fill_between(test_age,local_nl_mean.detach().numpy()-2*local_nl_std.detach().numpy(),local_nl_mean.detach().numpy()+2*local_nl_std.detach().numpy(),color='C3',alpha=0.3,label='95% CI')
        plt.xlabel('Age (BP)')
        plt.ylabel('RSL (m)')
        plt.legend()
        plt.xlim(-100,12000)

        plt.subplot(256)
        rsl_time,rsl_rate,rsl_rate_sd = PSTHM.post.cal_rate_var(test_age,y_var.detach().numpy(),y_mean.detach().numpy())
        all_total_rate_mean[i] = rsl_rate
        all_total_rate_std[i] = rsl_rate_sd
        plt.plot(rsl_time,-rsl_rate*1000,'C5',linewidth=3,label='Total rate')
        plt.fill_between(rsl_time,-(rsl_rate-2*rsl_rate_sd)*1000,-(rsl_rate+2*rsl_rate_sd)*1000,color='C5',alpha=0.3,label='95% CI')
        plt.xlabel('Age (BP)')
        plt.ylabel('Rate (mm/yr)')
        plt.legend()
        plt.xlim(-100,12000)
        plt.subplot(257)
        _,rsl_rate_global,rsl_rate_sd_global = PSTHM.post.cal_rate_var(test_age,global_var.detach().numpy(),global_mean.detach().numpy())
        plt.plot(rsl_time,-rsl_rate_global*1000,'C0',linewidth=3,label='Global rate')
        plt.fill_between(rsl_time,-(rsl_rate_global-2*rsl_rate_sd_global)*1000,-(rsl_rate_global+2*rsl_rate_sd_global)*1000,color='C0',alpha=0.3,label='95% CI')
        plt.xlabel('Age (BP)')
        plt.ylabel('Rate (mm/yr)')
        plt.legend()
        plt.xlim(-100,12000)
        plt.subplot(258)
        _,rsl_rate_regional_linear,rsl_rate_sd_regional_linear = PSTHM.post.cal_rate_var(test_age,regional_linear_var.detach().numpy(),regional_linear_mean.detach().numpy())
        plt.plot(rsl_time,-rsl_rate_regional_linear*1000,'C1',linewidth=3,label='Regional Linear rate')
        plt.fill_between(rsl_time,-(rsl_rate_regional_linear-2*rsl_rate_sd_regional_linear)*1000,-(rsl_rate_regional_linear+2*rsl_rate_sd_regional_linear)*1000,color='C1',alpha=0.3,label='95% CI')
        plt.xlabel('Age (BP)')
        plt.ylabel('Rate (mm/yr)')
        plt.legend()
        plt.xlim(-100,12000)
        plt.subplot(259)
        _,rsl_rate_regional_nl,rsl_rate_sd_regional_nl = PSTHM.post.cal_rate_var(test_age,regional_nl_var.detach().numpy(),regional_nl_mean.detach().numpy())
        plt.plot(rsl_time,-rsl_rate_regional_nl*1000,'C2',linewidth=3,label='Regional Nonlinear rate')
        plt.fill_between(rsl_time,-(rsl_rate_regional_nl-2*rsl_rate_sd_regional_nl)*1000,-(rsl_rate_regional_nl+2*rsl_rate_sd_regional_nl)*1000,color='C2',alpha=0.3,label='95% CI')
        plt.xlabel('Age (BP)')
        plt.ylabel('Rate (mm/yr)')
        plt.legend()
        plt.xlim(-100,12000)
        plt.subplot(2,5,10)
        _,rsl_rate_local_nl,rsl_rate_sd_local_nl = PSTHM.post.cal_rate_var(test_age,local_nl_var.detach().numpy(),local_nl_mean.detach().numpy())
        plt.plot(rsl_time,-rsl_rate_local_nl*1000,'C3',linewidth=3,label='Local Nonlinear rate')
        plt.fill_between(rsl_time,-(rsl_rate_local_nl-2*rsl_rate_sd_local_nl)*1000,-(rsl_rate_local_nl+2*rsl_rate_sd_local_nl)*1000,color='C3',alpha=0.3,label='95% CI')
        plt.xlabel('Age (BP)')
        plt.ylabel('Rate (mm/yr)')
        plt.legend()
        plt.suptitle(all_region_name[i],fontsize=30)
        # plt.tight_layout()
        plt.xlim(-100,12000)
        all_total_mean[i] = y_mean.detach().numpy()
        all_total_std[i] = y_std.detach().numpy()
        if i==0:
            all_global_mean = global_mean.detach().numpy()
            all_global_std = global_std.detach().numpy()
        all_regional_linear_mean[i] = regional_linear_mean.detach().numpy()
        all_regional_linear_std[i] = regional_linear_std.detach().numpy()
        all_regional_nl_mean[i] = regional_nl_mean.detach().numpy()
        all_regional_nl_std[i] = regional_nl_std.detach().numpy()
        all_local_nl_mean[i] = local_nl_mean.detach().numpy()
        all_local_nl_std[i] = local_nl_std.detach().numpy()
        if i==0:
            all_global_rate_mean = rsl_rate_global
            all_global_rate_std = rsl_rate_sd_global
        all_regional_linear_rate_mean[i] = rsl_rate_regional_linear
        all_regional_linear_rate_std[i] = rsl_rate_sd_regional_linear
        all_regional_nl_rate_mean[i] = rsl_rate_regional_nl
        all_regional_nl_rate_std[i] = rsl_rate_sd_regional_nl
        all_local_nl_rate_mean[i] = rsl_rate_local_nl
        all_local_nl_rate_std[i] = rsl_rate_sd_local_nl
        #check if the directory exists
        if not os.path.exists('Outputs/Temporal_plots_data/'):
            os.makedirs('Outputs/Temporal_plots_data/')
        plt.savefig('Outputs/Temporal_plots_data/'+all_region_name[i]+'.pdf',dpi=200);
        plt.clf()
    print('A total number of '+str(len(all_region))+' figures have been generated and saved to Outputs/Temporal_plots_data/')
    # Create a NetCDF file
    nc_filename = 'Outputs/Temporal_plots_data/output_results.nc'
    with nc.Dataset(nc_filename, 'w', format='NETCDF4') as dataset:
        # Define dimensions
        region_dim = dataset.createDimension('region', len(all_region))
        time_dim = dataset.createDimension('time', len(test_age))
        rate_time_dim = dataset.createDimension('rate_time', len(test_age)-1)
        
        # Define variables
        regions = dataset.createVariable('regions', 'S1', ('region',))
        time = dataset.createVariable('time', 'f4', ('time',))
        rate_time = dataset.createVariable('rate_time', 'f4', ('rate_time',))

        total_mean = dataset.createVariable('total_mean', 'f4', ('region', 'time'))
        total_std = dataset.createVariable('total_std', 'f4', ('region', 'time'))
        global_mean = dataset.createVariable('global_mean', 'f4', ('time',))
        global_std = dataset.createVariable('global_std', 'f4', ('time',))
        regional_linear_mean = dataset.createVariable('regional_linear_mean', 'f4', ('region', 'time'))
        regional_linear_std = dataset.createVariable('regional_linear_std', 'f4', ('region', 'time'))
        regional_nl_mean = dataset.createVariable('regional_nl_mean', 'f4', ('region', 'time'))
        regional_nl_std = dataset.createVariable('regional_nl_std', 'f4', ('region', 'time'))
        local_nl_mean = dataset.createVariable('local_nl_mean', 'f4', ('region', 'time'))
        local_nl_std = dataset.createVariable('local_nl_std', 'f4', ('region', 'time'))

        total_rate_mean = dataset.createVariable('total_rate_mean', 'f4', ('region', 'rate_time'))
        total_rate_std = dataset.createVariable('total_rate_std', 'f4', ('region', 'rate_time'))
        global_rate_mean = dataset.createVariable('global_rate_mean', 'f4', ('rate_time',))
        global_rate_std = dataset.createVariable('global_rate_std', 'f4', ('rate_time',))
        regional_linear_rate_mean = dataset.createVariable('regional_linear_rate_mean', 'f4',  ('region', 'rate_time'))
        regional_linear_rate_std = dataset.createVariable('regional_linear_rate_std', 'f4',  ('region', 'rate_time'))
        regional_nl_rate_mean = dataset.createVariable('regional_nl_rate_mean', 'f4',  ('region', 'rate_time'))
        regional_nl_rate_std = dataset.createVariable('regional_nl_rate_std', 'f4',  ('region', 'rate_time'))
        local_nl_rate_mean = dataset.createVariable('local_nl_rate_mean', 'f4',  ('region', 'rate_time'))
        local_nl_rate_std = dataset.createVariable('local_nl_rate_std', 'f4',  ('region', 'rate_time'))

        # Assign data to variables
        regions[:] = np.array([region for region in all_region_name], dtype='S1')  # Convert region names
        time[:] = test_age
        rate_time[:] = test_age[:-1]  # For rate time, use one fewer time step

        total_mean[:, :] = all_total_mean
        total_std[:, :] = all_total_std
        global_mean[:] = all_global_mean
        global_std[:] = all_global_std
        regional_linear_mean[:, :] = all_regional_linear_mean
        regional_linear_std[:, :] = all_regional_linear_std
        regional_nl_mean[:, :] = all_regional_nl_mean
        regional_nl_std[:, :] = all_regional_nl_std
        local_nl_mean[:, :] = all_local_nl_mean
        local_nl_std[:, :] = all_local_nl_std

        total_rate_mean[:, :] = all_total_rate_mean
        total_rate_std[:, :] = all_total_rate_std
        global_rate_mean[:] = all_global_rate_mean
        global_rate_std[:] = all_global_rate_std
        regional_linear_rate_mean[:, :] = all_regional_linear_rate_mean
        regional_linear_rate_std[:, :] = all_regional_linear_rate_std
        regional_nl_rate_mean[:, :] = all_regional_nl_rate_mean
        regional_nl_rate_std[:, :] = all_regional_nl_rate_std
        local_nl_rate_mean[:, :] = all_local_nl_rate_mean
        local_nl_rate_std[:, :] = all_local_nl_rate_std

    print(f"Temporal analysis data has been successfully saved to {nc_filename}")


def process_rsl_predictions_and_save(rsl_lat, rsl_lon, gpr, test_time=np.arange(12000, 0, -1000), lat_step=0.3, lon_step=0.3, filename='Outputs/Spatial_plots_data/output_results.nc'):
    """
    A function to process the RSL predictions, decompose the kernels, generate plots, and save results into a NetCDF file.
    
    Parameters:
    - rsl_lat: Array of latitudes for the RSL data.
    - rsl_lon: Array of longitudes for the RSL data.
    - gpr: Gaussian Process Regression model.
    - test_time: Array of time values to process (default: np.arange(12000, 0, -1000)).
    - lat_step: Step size for latitude in the prediction grid (default: 0.3).
    - lon_step: Step size for longitude in the prediction grid (default: 0.3).
    - filename: Path to save the NetCDF file (default: 'Outputs/Spatial_plots_data/output_results.nc').
    """
    # Generate prediction grid
    lat_matrix = np.arange(np.min(rsl_lat) - 2, np.max(rsl_lat) + 2, lat_step)
    lon_matrix = np.arange(np.min(rsl_lon) - 2, np.max(rsl_lon) + 2, lon_step)
    pred_matrix = PSTHM.post.gen_pred_matrix(test_time[0], lat_matrix, lon_matrix)

    # Coordinates of RSL data
    rsl_data_coord = np.array([rsl_lat, rsl_lon]).T
    select_index = []
    for i in range(len(pred_matrix)):
        dis = np.sum(np.abs(pred_matrix[i, 1:].detach().numpy() - rsl_data_coord) ** 2, axis=1) ** 0.5
        if np.min(dis) < 2:
            select_index.append(i)
    select_index = np.array(select_index)
    pred_matrix = pred_matrix[select_index]

    # Initialize arrays to store results
    all_total_mean = np.zeros([len(test_time), len(pred_matrix)])
    all_total_std = np.zeros([len(test_time), len(pred_matrix)])
    all_global_mean = np.zeros([len(test_time), len(pred_matrix)])
    all_global_std = np.zeros([len(test_time), len(pred_matrix)])
    all_regional_nl_mean = np.zeros([len(test_time), len(pred_matrix)])
    all_regional_nl_std = np.zeros([len(test_time), len(pred_matrix)])
    all_regional_linear_mean = np.zeros([len(test_time), len(pred_matrix)])
    all_regional_linear_std = np.zeros([len(test_time), len(pred_matrix)])
    all_local_nl_mean = np.zeros([len(test_time), len(pred_matrix)])
    all_local_nl_std = np.zeros([len(test_time), len(pred_matrix)])

    latitudes = pred_matrix[:, 1]
    longitudes = pred_matrix[:, 2]

    # Define the kernels for decomposition
    test_global_kernel = gpr.kernel.kern0.kern0.kern0.kern0.kern0
    regional_linear_kernel = gpr.kernel.kern0.kern0.kern0.kern0.kern1
    regional_nl_kernel = gpr.kernel.kern0.kern0.kern0.kern1
    local_nl_kernel = gpr.kernel.kern0.kern0.kern1

    for i in range(len(test_time)):
        pred_matrix[:, 0] = test_time[i]
        y_mean, y_var = gpr(pred_matrix, full_cov=True)
        y_std = y_var.diag().sqrt()

        # Decompose the kernel contributions
        global_dep, regional_linear_dep, regional_nl_dep, local_nl_dep = PSTHM.post.decompose_kernels(
            gpr, pred_matrix, [test_global_kernel, regional_linear_kernel, regional_nl_kernel, local_nl_kernel])

        global_mean, global_var = global_dep
        global_std = global_var.diag().sqrt()

        regional_linear_mean, regional_linear_var = regional_linear_dep
        regional_linear_std = regional_linear_var.diag().sqrt()

        regional_nl_mean, regional_nl_var = regional_nl_dep
        regional_nl_std = regional_nl_var.diag().sqrt()

        local_nl_mean, local_nl_var = local_nl_dep
        local_nl_std = local_nl_var.diag().sqrt()

        # Store the results for this time step
        all_total_mean[i] = y_mean.detach().numpy()
        all_total_std[i] = y_std.detach().numpy()
        all_global_mean[i] = global_mean.detach().numpy()
        all_global_std[i] = global_std.detach().numpy()
        all_regional_nl_mean[i] = regional_nl_mean.detach().numpy()
        all_regional_nl_std[i] = regional_nl_std.detach().numpy()
        all_regional_linear_mean[i] = regional_linear_mean.detach().numpy()
        all_regional_linear_std[i] = regional_linear_std.detach().numpy()
        all_local_nl_mean[i] = local_nl_mean.detach().numpy()
        all_local_nl_std[i] = local_nl_std.detach().numpy()

        # ------------------ Plot the results ------------------
        fig = plt.figure(figsize=(50, 20))

        # Total Mean Prediction
        ax = plt.subplot(251, projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.LAND, edgecolor='black', zorder=0, alpha=0.5)
        plt.scatter(pred_matrix[:, 2], pred_matrix[:, 1], c=y_mean.detach().numpy(), s=200, marker='s', zorder=-1, vmax=0, vmin=-25,cmap='turbo')
        plt.title('Total Mean Prediction (m)')
        plt.colorbar()

        # Global kernel contribution
        ax = plt.subplot(252, projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.LAND, edgecolor='black', zorder=0, alpha=0.5)
        plt.scatter(pred_matrix[:, 2], pred_matrix[:, 1], c=global_mean.detach().numpy(), s=200, marker='s', zorder=-1, vmax=5, vmin=-30)
        plt.title('Global kernel contribution (m)')
        plt.colorbar()

        # Regional linear kernel contribution
        ax = plt.subplot(253, projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.LAND, edgecolor='black', zorder=0, alpha=0.5)
        plt.scatter(pred_matrix[:, 2], pred_matrix[:, 1], c=regional_linear_mean.detach().numpy(), s=200, marker='s', zorder=-1, vmax=10, vmin=-10, cmap='PuOr_r')
        plt.title('Regional linear kernel contribution (m)')
        plt.colorbar()

        # Regional non-linear kernel contribution
        ax = plt.subplot(254, projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.LAND, edgecolor='black', zorder=0, alpha=0.5)
        plt.scatter(pred_matrix[:, 2], pred_matrix[:, 1], c=regional_nl_mean.detach().numpy(), s=200, marker='s', zorder=-1, vmax=0.5, vmin=-0.5, cmap='RdBu_r')
        plt.title('Regional non-linear kernel contribution (m)')
        plt.colorbar()

        # Local non-linear kernel contribution
        ax = plt.subplot(255, projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.LAND, edgecolor='black', zorder=0, alpha=0.5)
        plt.scatter(pred_matrix[:, 2], pred_matrix[:, 1], c=local_nl_mean.detach().numpy(), s=200, marker='s', zorder=-1, vmax=0.5, vmin=-0.5, cmap='PRGn_r')
        plt.title('Local non-linear kernel contribution (m)')
        plt.colorbar()

        # Standard deviation plots
        all_std = np.array([y_std.detach().numpy(), global_std.detach().numpy(), regional_linear_std.detach().numpy(),
                             regional_nl_std.detach().numpy(), local_nl_std.detach().numpy()]).flatten()
        std_vmin = np.min(all_std)
        std_vmax = np.max(all_std)

        ax = plt.subplot(256, projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.LAND, edgecolor='black', zorder=0, alpha=0.5)
        plt.scatter(pred_matrix[:, 2], pred_matrix[:, 1], c=y_std.detach().numpy(), s=200, marker='s', zorder=-1, cmap='magma', vmin=std_vmin, vmax=std_vmax)
        plt.title('Total Mean standard deviation (m)')
        plt.colorbar()

        ax = plt.subplot(257, projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.LAND, edgecolor='black', zorder=0, alpha=0.5)
        plt.scatter(pred_matrix[:, 2], pred_matrix[:, 1], c=global_std.detach().numpy(), s=200, marker='s', zorder=-1, cmap='magma', vmin=std_vmin, vmax=std_vmax)
        plt.title('Global kernel standard deviation (m)')
        plt.colorbar()

        ax = plt.subplot(258, projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.LAND, edgecolor='black', zorder=0, alpha=0.5)
        plt.scatter(pred_matrix[:, 2], pred_matrix[:, 1], c=regional_linear_std.detach().numpy(), s=200, marker='s', zorder=-1, cmap='magma', vmin=std_vmin, vmax=std_vmax)
        plt.title('Regional linear kernel standard deviation (m)')
        plt.colorbar()

        ax = plt.subplot(259, projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.LAND, edgecolor='black', zorder=0, alpha=0.5)
        plt.scatter(pred_matrix[:, 2], pred_matrix[:, 1], c=regional_nl_std.detach().numpy(), s=200, marker='s', zorder=-1, cmap='magma', vmin=std_vmin, vmax=std_vmax)
        plt.title('Regional non-linear kernel standard deviation (m)')
        plt.colorbar()

        ax = plt.subplot(2, 5, 10, projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.LAND, edgecolor='black', zorder=0, alpha=0.5)
        plt.scatter(pred_matrix[:, 2], pred_matrix[:, 1], c=local_nl_std.detach().numpy(), s=200, marker='s', zorder=-1, cmap='magma', vmin=std_vmin, vmax=std_vmax)
        plt.title('Local non-linear kernel standard deviation (m)')
        plt.colorbar()
        #check if the directory exists
        if not os.path.exists('Outputs/Spatial_plots_data'):
            os.makedirs('Outputs/Spatial_plots_data')
        plt.savefig(f'Outputs/Spatial_plots_data/{test_time[i]}.pdf', dpi=300)
        plt.clf()
    print('A total of', len(test_time), 'plots have been saved in the Outputs/Spatial_plots_data directory.')
    # ----------------- Save results to NetCDF file -----------------
    with nc.Dataset(filename, 'w', format='NETCDF4') as dataset:
        time_dim = dataset.createDimension('time', len(test_time))
        point_dim = dataset.createDimension('point', len(pred_matrix))

        times = dataset.createVariable('time', 'f4', ('time',))
        latitudes_var = dataset.createVariable('latitude', 'f4', ('point',))
        longitudes_var = dataset.createVariable('longitude', 'f4', ('point',))

        total_mean_var = dataset.createVariable('total_mean', 'f4', ('time', 'point'))
        total_std_var = dataset.createVariable('total_std', 'f4', ('time', 'point'))
        global_mean_var = dataset.createVariable('global_mean', 'f4', ('time', 'point'))
        global_std_var = dataset.createVariable('global_std', 'f4', ('time', 'point'))
        regional_nl_mean_var = dataset.createVariable('regional_nl_mean', 'f4', ('time', 'point'))
        regional_nl_std_var = dataset.createVariable('regional_nl_std', 'f4', ('time', 'point'))
        regional_linear_mean_var = dataset.createVariable('regional_linear_mean', 'f4', ('time', 'point'))
        regional_linear_std_var = dataset.createVariable('regional_linear_std', 'f4', ('time', 'point'))
        local_nl_mean_var = dataset.createVariable('local_nl_mean', 'f4', ('time', 'point'))
        local_nl_std_var = dataset.createVariable('local_nl_std', 'f4', ('time', 'point'))

        # Assign data to NetCDF variables
        times[:] = test_time
        latitudes_var[:] = latitudes
        longitudes_var[:] = longitudes
        total_mean_var[:, :] = all_total_mean
        total_std_var[:, :] = all_total_std
        global_mean_var[:, :] = all_global_mean
        global_std_var[:, :] = all_global_std
        regional_nl_mean_var[:, :] = all_regional_nl_mean
        regional_nl_std_var[:, :] = all_regional_nl_std
        regional_linear_mean_var[:, :] = all_regional_linear_mean
        regional_linear_std_var[:, :] = all_regional_linear_std
        local_nl_mean_var[:, :] = all_local_nl_mean
        local_nl_std_var[:, :] = all_local_nl_std

    print('Spatial analysis data has been successfully saved to Outputs/Spatial_plots_data/output_results.nc')
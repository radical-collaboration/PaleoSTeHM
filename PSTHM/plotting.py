#----------------------Define Functions---------------------------
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from scipy import interpolate
import torch
font = {'weight':'normal',
       'size':20,
       'family':'Helvetica'}
matplotlib.rcParams['xtick.major.size'] = 8
matplotlib.rcParams['ytick.major.size'] = 8
matplotlib.rcParams['axes.labelsize'] = 22
matplotlib.rcParams['figure.figsize'] = (12, 6)
matplotlib.rcParams['legend.frameon'] = 'False'
matplotlib.rc('font',**font)



def plot_uncertainty_boxes( x, y, x_error, y_error, ax=None):
    '''
    A function to plot uncertainty box for data with vertical and horizontal uncertainties.

    -----------------Input-------------------
    x: a 1-D array of input data
    y: a 1-D array of ouput data
    x_error: a 1-D array containing 2 sigma uncertainty of input data
    y_error: a 1-D array containing 2 sigma uncertainty of output data

    ---------------Return-------------------
    ax: a matplotlib ax of output plot
    '''
    if ax == None: ax = plt.subplot(111)
    for i in range(len(x)):
        ax.add_patch(plt.Rectangle((x[i] - x_error[i], y[i] - y_error[i]), 2 * x_error[i], 2*  y_error[i], 
                                    fill=True, fc=(1,0.82,0.86,0.15),ec=(0.7,0,0,0.7), linewidth=3))

    #     ax.set_xlim(np.min(x)-x_error[np.argmin(x)]*5,np.max(x)+x_error[np.argmax(x)]*5)
    #     ax.set_ylim(np.min(y)-y_error[np.argmin(y)]*5,np.max(y)+y_error[np.argmax(y)]*5)

    ax.set_xlabel('Age (CE)')
    ax.set_ylabel('RSL (m)')

    return ax

def plot_tem_regression(data_age, data_rsl, data_age_sigma, data_rsl_sigma, mean_rsl_age, mean_rsl,
                        rsl_sd, rsl_rate_age, mean_rate, rate_sd, color='C0', axes=None):
    '''
    A function to create matplotlib plot for temporal regression results.

    -----------------Inputs-------------------
    data_age: a 1-D array of rsl age data
    data_rsl: a 1-D array of rsl data
    data_age_sigma: a 1-D array containing 1 sigma uncertainty of rsl age data
    data_rsl_sigma: a 1-D array containing 1 sigma uncertainty of rsl data
    mean_rsl_age: a 1-D array of testing age data, i.e., new_X for GP regression
    mean_rsl: a 1-D array of mean rsl regression results
    rsl_sd: a 1-D array of rsl regression standard deviation
    rsl_rate_age: a 1-D array of testing age data for rsl rate
    rsd_rate: a 1-D array of mean rsl rate regression results
    rate_sd: a 1-D array of rsl rate regression standard deviation
    axes: matplotlib axes, if None, create a new figure
    save: bool, whether to save the plot

    ---------------Output-------------------
    A matplotlib plot of temporal regression results which contains
    three sub-plots: 1) RSL data with uncertainty box and mean regression line; 2) RSL rate
    with uncertainty box and mean regression line; 3) Residual plot.


    '''
    # change torch tensor to numpy array for plotting
    if torch.is_tensor(data_age) == True: data_age = data_age.detach().numpy()
    if torch.is_tensor(data_rsl) == True: data_rsl = data_rsl.detach().numpy()
    if torch.is_tensor(data_age_sigma) == True: data_age_sigma = data_age_sigma.detach().numpy()
    if torch.is_tensor(data_rsl_sigma) == True: data_rsl_sigma = data_rsl_sigma.detach().numpy()
    if torch.is_tensor(mean_rsl_age) == True: mean_rsl_age = mean_rsl_age.detach().numpy()
    if torch.is_tensor(mean_rsl) == True: mean_rsl = mean_rsl.detach().numpy()
    if torch.is_tensor(rsl_sd) == True: rsl_sd = rsl_sd.detach().numpy()
    if torch.is_tensor(rsl_rate_age) == True: rsl_rate_age = rsl_rate_age.detach().numpy()
    if torch.is_tensor(mean_rate) == True: mean_rate = mean_rate.detach().numpy()
    if torch.is_tensor(rate_sd) == True: rate_sd = rate_sd.detach().numpy()

    if axes == None:
        fig, axes = plt.subplots(nrows=1,
                                    ncols=3,
                                    figsize=(36, 10)
                                    )
    ax = axes[0]

    plot_uncertainty_boxes(data_age,
                            data_rsl,
                            data_age_sigma * 2,
                            data_rsl_sigma * 2,
                            ax=ax
                            )

    ax.plot(mean_rsl_age,
            mean_rsl,
            linewidth=3,
            label='Mean'
            )

    ax.fill_between(
        mean_rsl_age,  # plot the two-sigma uncertainty about the mean
        (mean_rsl - 2.0 * rsl_sd),
        (mean_rsl + 2.0 * rsl_sd),
        color=color,
        alpha=0.6, zorder=10, label='95% CI')

    ax.legend(loc=0)
    ax = axes[1]

    ax.plot(rsl_rate_age,
            mean_rate * 1000,
            linewidth=3,
            label='Mean'
            )

    ax.fill_between(
        rsl_rate_age,  # plot the two-sigma uncertainty about the mean
        (mean_rate - 2.0 * rate_sd) * 1000,
        (mean_rate + 2.0 * rate_sd) * 1000,
        color=color,
        alpha=0.6,
        zorder=10,
        label='95% CI'
    )

    ax.set_xlabel('Age (CE)')
    ax.set_ylabel('RSL rate (mm/year)')
    ax.legend(loc=0)
    ax = axes[2]

    f = interpolate.interp1d(mean_rsl_age,
                                mean_rsl
                                )

    ax.scatter(data_age,
                (data_rsl - f(data_age)) * 1000,
                s=150, marker='*',
                color=color,
                alpha=0.6
                )

    ax.set_xlabel('Age (CE)')
    ax.set_ylabel('Residual (mm)');
    plt.show()

    return axes

def plot_loss(loss):
    '''A function used to plot loss function variation'''
    plt.plot(loss)
    plt.xlabel("Iterations")
    _ = plt.ylabel("Loss")  # supress output text

def plot_spatial_rsl_single(pred_matrix, y_mean, y_var, cmap='viridis'):
    '''
    A function to plot the spatial RSL map and uncertainty map

    ------Inputs------
    pred_matrix: a matrix with 3 columns, the first column is the age, the second column is the latitude, the third column is the longitude
    y_mean: the mean of the predicted RSL
    y_var: the covariance matrix of the predicted RSL

    ------Outputs------
    A figure with two subplots, the left subplot is the RSL map, the right subplot is the RSL uncertainty map
    '''
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if torch.is_tensor(pred_matrix):
        pred_matrix = pred_matrix.detach().numpy()
    lat_matrix = np.unique(pred_matrix[:, 1])
    lon_matrix = np.unique(pred_matrix[:, 2])
    lon_mat, lat_mat = np.meshgrid(lon_matrix, lat_matrix)

    fig = plt.figure(figsize=(20, 10))

    ax2 = fig.add_subplot(1,
                          2,
                          1,
                          projection=ccrs.PlateCarree())

    ax2.add_feature(cartopy.feature.LAND,
                    edgecolor='black',
                    zorder=10,
                    alpha=0.5
                    )

    ax2.add_feature(cfeature.STATES,
                    edgecolor='black',
                    zorder=10
                    )

    ax2.set_extent([np.min(pred_matrix[:, 2]),
                    np.max(pred_matrix[:, 2]),
                    np.min(pred_matrix[:, 1]),
                    np.max(pred_matrix[:, 1])]
                    )

    cax = ax2.pcolor(lon_mat,
                        lat_mat, y_mean.detach().numpy().reshape(lon_mat.shape),
                        transform=ccrs.PlateCarree(),
                        cmap=cmap
                        )

    cbar = fig.colorbar(cax,
                        ax=ax2,
                        orientation='vertical',
                        pad=0.01
                        )

    cbar.set_label('RSL (m)')
    ax2.set_title('{:5.1f} CE'.format(pred_matrix[0, 0]))

    ax2 = fig.add_subplot(1,
                          2,
                          2,
                          projection=ccrs.PlateCarree())

    ax2.set_extent([np.min(pred_matrix[:, 2]),
                    np.max(pred_matrix[:, 2]),
                    np.min(pred_matrix[:, 1]),
                    np.max(pred_matrix[:, 1])]
                    )

    ax2.add_feature(cartopy.feature.LAND,
                    edgecolor='black',
                    zorder=10,
                    alpha=0.5
                    )

    ax2.add_feature(cfeature.STATES,
                    edgecolor='black',
                    zorder=10
                    )

    y_std = y_var.diag().sqrt()

    cax = ax2.pcolor(lon_mat,
                        lat_mat,
                        y_std.detach().numpy().reshape(lon_mat.shape),
                        transform=ccrs.PlateCarree(),
                        cmap=cmap
                        )

    cbar = fig.colorbar(cax,
                        ax=ax2,
                        orientation='vertical',
                        pad=0.01
                        )

    cbar.set_label('RSL uncertainty (m)')
    ax2.set_title('{:5.1f} CE'.format(pred_matrix[0, 0]));

def plot_spatial_rsl_range(pred_matrix, y_mean, y_var, rsl_lon, rsl_lat, rsl_age, rsl_region, cmap='viridis',
                            plot_site=False):
    '''
    A function to plot the spatial mean RSL, RSL change rate and RSL uncertainty maps

    --------Inputs--------
    pred_matrix: a matrix with 3 columns, the first column is the age, the second column is the latitude, the third column is the longitude
    y_mean: the mean of the predicted RSL from GP model
    y_var: the covariance matrix of the predicted RSL from GP model
    rsl_lon: the longitude of the RSL data
    rsl_lat: the latitude of the RSL data
    rsl_age: the age of the RSL data
    rsl_region: the region of the RSL data
    cmap: the colormap used in the plot

    --------Outputs--------
    A figure with three subplots, the left subplot is the mean RSL map, the middle subplot is the RSL change rate map,
    the right subplot is the RSL uncertainty map
    '''
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if torch.is_tensor(pred_matrix):
        pred_matrix = pred_matrix.detach().numpy()

    time_mat = np.unique(pred_matrix[:, 0])
    lon_matrix = np.unique(pred_matrix[:, 2])
    lat_matrix = np.unique(pred_matrix[:, 1])

    lon_mat, lat_mat = np.meshgrid(lon_matrix,
                                    lat_matrix
                                    )
    y_std = y_var.diag().sqrt()

    mean_rsl = np.zeros([len(lat_matrix),
                            len(lon_matrix)]
                        )

    for i in range(len(time_mat)):
        mean_rsl += y_mean[i::len(time_mat)].reshape([len(lat_matrix), len(lon_matrix)]).detach().numpy()
    mean_rsl = mean_rsl / len(time_mat)

    min_time = np.min(time_mat)
    max_time = np.max(time_mat)

    fig = plt.figure(figsize=(30, 10))
    # -----------------plot the mean RSL map-----------------
    ax2 = fig.add_subplot(1,
                          3,
                          1,
                          projection=ccrs.PlateCarree())

    ax2.add_feature(cartopy.feature.LAND,
                    edgecolor='black',
                    zorder=10,
                    alpha=0.5
                    )

    ax2.add_feature(cfeature.STATES,
                    edgecolor='black',
                    zorder=10
                    )

    ax2.set_extent([np.min(lon_matrix),
                    np.max(lon_matrix),
                    np.min(lat_matrix),
                    np.max(lat_matrix)]
                    )

    cax = ax2.pcolor(lon_mat,
                        lat_mat, mean_rsl,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap
                        )

    cbar = fig.colorbar(cax,
                        ax=ax2,
                        orientation='vertical',
                        pad=0.01
                        )

    cbar.set_label('RSL (m)')
    ax2.set_title('{:5.1f} to {:5.1f} CE'.format(min_time, max_time))

    # -----------------plot the RSL rate map-----------------
    rsl_rate = (y_mean[0::len(time_mat)] - y_mean[len(time_mat) - 1::len(time_mat)]).detach().numpy().reshape(
        [len(lat_matrix), len(lon_matrix)]) / (time_mat[0] - time_mat[-1])

    ax2 = fig.add_subplot(1,
                          3,
                          2,
                          projection=ccrs.PlateCarree())

    ax2.set_extent([np.min(pred_matrix[:, 2]),
                    np.max(pred_matrix[:, 2]),
                    np.min(pred_matrix[:, 1]),
                    np.max(pred_matrix[:, 1])]
                    )

    ax2.add_feature(cartopy.feature.LAND,
                    edgecolor='black',
                    zorder=10, alpha=0.5
                    )

    ax2.add_feature(cfeature.STATES,
                    edgecolor='black',
                    zorder=10
                    )

    cax = ax2.pcolor(lon_mat,
                        lat_mat,
                        rsl_rate.reshape(lon_mat.shape) * 1000,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap
                        )

    cbar = fig.colorbar(cax,
                        ax=ax2,
                        orientation='vertical',
                        pad=0.01
                        )

    cbar.set_label('RSL change rate (m/kyr)')
    ax2.set_title('{:5.1f} to {:5.1f} CE'.format(min_time, max_time))

    # -----------------plot the RSL rate map-----------------
    time_index = (rsl_age >= min_time) & (rsl_age <= max_time)

    ax2 = fig.add_subplot(1,
                          3,
                          3,
                          projection=ccrs.PlateCarree())

    sd_rsl = np.zeros([len(lat_matrix),
                        len(lon_matrix)]
                        )

    for i in range(len(time_mat)):
        sd_rsl += y_std[i::len(time_mat)].reshape([len(lat_matrix), len(lon_matrix)]).detach().numpy()
    sd_rsl = sd_rsl / len(time_mat)

    ax2.set_extent([np.min(pred_matrix[:, 2]),
                    np.max(pred_matrix[:, 2]),
                    np.min(pred_matrix[:, 1]),
                    np.max(pred_matrix[:, 1])]
                    )

    ax2.add_feature(cartopy.feature.LAND,
                    edgecolor='black',
                    zorder=10,
                    alpha=0.5
                    )

    ax2.add_feature(cfeature.STATES,
                    edgecolor='black',
                    zorder=10
                    )

    cax = ax2.pcolor(lon_mat,
                        lat_mat,
                        sd_rsl,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap
                        )

    cbar = fig.colorbar(cax,
                        ax=ax2,
                        orientation='vertical',
                        pad=0.01
                        )

    cbar.set_label('One sigma RSL uncertainty (m)')

    if plot_site:
        for i in np.unique(rsl_region):
            region_index = rsl_region[time_index] == i

            ax2.scatter(np.mean(rsl_lon[time_index][region_index]),
                        np.mean(rsl_lat[time_index][region_index]),
                        transform=ccrs.PlateCarree(),
                        s=len(rsl_lon[time_index][region_index]) * 40,
                        marker='o',
                        facecolor='none',
                        ec='darkred',
                        linewidth=3,
                        zorder=20
                        )

        sc = ax2.scatter([0],
                            [0],
                            s=200,
                            label='5 RSL data',
                            marker='o',
                            facecolor='none',
                            ec='darkred',
                            zorder=-20,
                            linewidth=3
                            )

        sc2 = ax2.scatter([0],
                            [0],
                            s=400,
                            label='10 RSL data',
                            marker='o',
                            facecolor='none',
                            ec='darkred',
                            zorder=-20,
                            linewidth=3
                            )

        sc3 = ax2.scatter([0],
                            [0],
                            s=800,
                            label='20 RSL data',
                            marker='o',
                            facecolor='none',
                            ec='darkred',
                            zorder=-20,
                            linewidth=3)

        ax2.legend(handles=[sc, sc2, sc3],
                    labels=['5 RSL data', '10 RSL data', '20 RSL data'],
                    loc=4
                    )

    ax2.set_title('{:5.1f} to {:5.3f} CE'.format(min_time, max_time));

    return fig

def plot_track_list(track_list):
    '''
    A function to plot the track_list generated from SVI_optm function

    ------Inputs------
    track_list: a dictionary containing the name and values of tracking variables
    '''

    if track_list.shape[1] % 3 == 0:
        row_num = (track_list.shape[1]) // 3
    else:
        row_num = track_list.shape[1] // 3 + 1

    fig, axes = plt.subplots(row_num,
                                3,
                                figsize=(30, row_num * 8)
                                )

    if row_num == 1:
        for i in range(row_num * 3):
            axes[i].plot(np.arange(len(track_list)),
                            track_list.iloc[:, i]
                            )

            axes[i].set_title('{} : {:6.6f}'.format(track_list.columns[i]
                                                    , track_list.iloc[-1, i]))

    else:
        for i in range(row_num):
            for j in range(3):
                if i * 3 + j < track_list.shape[1]:

                    axes[i, j].plot(np.arange(len(track_list)),
                                    track_list.iloc[:, i * 3 + j]
                                    )

                    axes[i, j].set_title('{}: {:6.6f}'.format(track_list.columns[i * 3 + j],
                                                                track_list.iloc[-1, i * 3 + j]))
                else:
                    axes[i, j].set_visible(False)

    return axes

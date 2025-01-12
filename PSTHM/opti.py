#----------------------Define Functions---------------------------
import torch
import pandas as pd
import pyro
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
from pyro.infer import SVI, Trace_ELBO
from tqdm.notebook import tqdm
import torch
from pyro.infer import MCMC, NUTS,HMC
from pyro.optim import Adam, ExponentialLR

def SVI_optm(gpr,num_iteration=500,lr=0.1,decay_r = 1,step_size=100,equal_kernels=None):
    '''
    A funciton to optimize the hyperparameters of a GP model using SVI

    ---------Inputs-----------
    gpr: a GP model defined by pyro GPR regression
    num_iteration: number of iterations for the optimization
    lr: learning rate for the optimization
    decay_r: decay rate for the learning rate
    step_size: step size for the learning rate to decay. 
    A step size of 100 with a decay rate of 0.9 means that the learning rate will be decrease 10% for every 100 steps.
    equal_kernels: a list of list of kernels that will be set to have the same hyperparameter. For example, if we want to use
    the same lengthscale for global kernel and regional non-linear kernel, you can set  equal_kernels = [['gpr.kern0.kern0.kern0.lengthscale',’gpr.kern0.kern1.lengthscale‘]].
    This also supports multiple kernel pairs.

    ---------Outputs-----------
    gpr: a GP model with optimized hyperparameters
    track: a dictionary of loss
    '''
    
    #clear the param store
    pyro.clear_param_store()
    #convert the model to double precision
    gpr = gpr.double()
    #define the optimiser
    optimizer = torch.optim.Adam(gpr.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay_r)

    #define the loss function
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    #do the optimisation
    track_list = []

    for i in tqdm(range(num_iteration)):
        
        scheduler.step()
        optimizer.zero_grad()
        loss = loss_fn(gpr.model, gpr.guide)
        loss.backward()
        optimizer.step()
        gpr.set_mode("guide")
        tem_para =  []

        if equal_kernels==None: 
            pass
        else:
            for kernels in equal_kernels:
                for i4, kernel in enumerate(kernels):
                    
                    # Check if the kernel is in the param store or if its '_map' version is present
                    if kernel in pyro.get_param_store().keys():
                        pass
                    elif kernel + '_map' in pyro.get_param_store().keys():
                        kernels[i4] = kernel + '_map'
                    else:
                        print(kernel, ' not in pyro storage')
                
                # After updating the kernels, set the first kernel to equal the others
                for i5 in range(1, len(kernels)):
                    print(i5, kernels[i5])
                    pyro.get_param_store()[kernels[i5]] = pyro.get_param_store()[kernels[0]]

        for i2 in pyro.get_param_store().values():
            if i2.numel()==1:
                tem_para.append(i2.item())
            else:
                for i3 in i2:
                    tem_para.append(i3.item())
        track_list.append([loss.item(),*tem_para])
    
    #generate columns names for the track list
    col_name = ['loss' ]

    for i in (dict(pyro.get_param_store()).keys()):
        if pyro.get_param_store()[i].numel() ==1:
            col_name.append(i[7:].replace('_map',''))
        else:
            for i2 in range(pyro.get_param_store()[i].numel()):
                col_name.append(i[7:].replace('_map','')+'_'+str(i2))
    #convert the track list to a dataframe
    track_list=pd.DataFrame(track_list,columns=col_name)

    return gpr,track_list


def SVI_NI_optm(gpr,x_sigma,num_iteration=500,lr=0.1,decay_r = 1,step_size=100,equal_kernels=None,gpu=False):
    '''
    A funciton to optimize the hyperparameters of a GP model using SVI

    ---------Inputs-----------
    gpr: a GP model defined by pyro GPR regression
    x_sigma: one sigma uncertainty for input data
    num_iteration: number of iterations for the optimization
    lr: learning rate for the optimization
    step_size: step size for the learning rate to decay. 
    decay_r: decay rate for the learning rate
    A step size of 100 with a decay rate of 0.9 means that the learning rate will be decrease 10% for every 100 steps.
    equal_kernels: a list of list of kernels that will be set to have the same hyperparameter. For example, if we want to use
    the same lengthscale for global kernel and regional non-linear kernel, you can set  equal_kernels = [['gpr.kern0.kern0.kern0.lengthscale',’gpr.kern0.kern1.lengthscale‘]].
    This also supports multiple kernel pairs.
    
    gpu: whether use gpu to accelerate training 
    ---------Outputs-----------
    gpr: a GP model with optimized hyperparameters
    track: a dictionary of loss
    '''
    
    #clear the param store
    pyro.clear_param_store()
    #convert the model to double precision
    gpr = gpr.double()
    #define the optimiser
    optimizer = torch.optim.Adam(gpr.parameters(), lr=lr)
    #define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay_r)
    #define the loss function
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    #do the optimisation
    track_list = []
    N = len(gpr.X)
    if gpr.noise.dim()==1:
        y_sigma = gpr.noise**0.5
    elif gpr.noise.dim()==2:
        y_sigma = gpr.noise.view(-1)[:: N + 1]**0.5


    for i in tqdm(range(num_iteration)):
        #update vertical noise based on gradient
        if gpu:
            x_test = torch.tensor(gpr.X.clone(),requires_grad=True).cuda()
        else:
            x_test = torch.tensor(gpr.X.clone(),requires_grad=True)
        y_mean, _ = gpr(x_test.double(), full_cov=False)
        y_mean.sum().backward(retain_graph=True)
        if gpu:
            y_rate = x_test.grad.cuda()
        else:
            y_rate = x_test.grad
        if y_rate.ndim>1: y_rate = y_rate[:,0]
        new_sigma = torch.sqrt((y_rate**2*(x_sigma)**2)+y_sigma**2)
        if gpr.noise.dim()==1:
            gpr.noise = torch.tensor(new_sigma**2)
        elif gpr.noise.dim()==2:
            gpr.noise.view(-1)[:: N + 1] = torch.tensor(new_sigma**2)
            
        scheduler.step()
        optimizer.zero_grad()
        loss = loss_fn(gpr.model, gpr.guide)
        loss.backward()
        optimizer.step()
        gpr.set_mode("guide")
        tem_para =  []
        
        if equal_kernels==None: 
            pass
        else:
            for kernels in equal_kernels:
                for i4, kernel in enumerate(kernels):
                    
                    # Check if the kernel is in the param store or if its '_map' version is present
                    if kernel in pyro.get_param_store().keys():
                        pass
                    elif kernel + '_map' in pyro.get_param_store().keys():
                        kernels[i4] = kernel + '_map'
                    else:
                        print(kernel, ' not in pyro storage')
                
                # After updating the kernels, set the first kernel to equal the others
                for i5 in range(1, len(kernels)):
                    print(i5, kernels[i5])
                    pyro.get_param_store()[kernels[i5]] = pyro.get_param_store()[kernels[0]]
        
        for i2 in pyro.get_param_store().values():
            if i2.numel()==1:
                tem_para.append(i2.item())
            else:
                for i3 in i2:
                    tem_para.append(i3.item())
        
        track_list.append([loss.item(),*tem_para])
    
    #generate columns names for the track list
    col_name = ['loss' ]

    for i in (dict(pyro.get_param_store()).keys()):
        if pyro.get_param_store()[i].numel() ==1:
            col_name.append(i[7:].replace('_map',''))
        else:
            for i2 in range(pyro.get_param_store()[i].numel()):
                col_name.append(i[7:].replace('_map','')+'_'+str(i2))
    #convert the track list to a dataframe
    track_list=pd.DataFrame(track_list,columns=col_name)

    return gpr,track_list


def NUTS_mcmc(gpr,num_samples=2500,warmup_steps=200,target_accept_prob = 0.8,print_stats=False):
    '''
    A function to run NUTS MCMC for GP regression model

    ----------Inputs---------
    gpr: a pyro GP regression model
    num_samples: number of samples to draw from the posterior
    warmup_steps: number of warmup steps for NUTS
    target_accept_prob: target acceptance probability for NUTS
    print_stats: whether to print the states of the model

    ----------Outputs---------
    mcmc: a pyro MCMC object
    
    '''
    hmc_kernel = NUTS(gpr.model,target_accept_prob=target_accept_prob)
    mcmc = MCMC(hmc_kernel, num_samples=num_samples,warmup_steps=warmup_steps)
    mcmc.run()
    if print_stats:
        for name, value in mcmc.get_samples().items():
            if 'kernel' in name:
                
                print('-----{}: {:4.2f} +/ {:4.2f} (2sd)-----'.format(name,value.mean(),2*value.std()))
                print('Gelman-Rubin statistic for {}: {:4.2f}'.format(name,mcmc.diagnostics()[name]['r_hat'].item()))
                print('Effective sample size for {}: {:4.2f}'.format(name,mcmc.diagnostics()[name]['n_eff'].item()))

    return mcmc

def HMC_mcmc(gpr,num_samples=2500,warmup_steps=200,target_accept_prob = 0.8,print_stats=False):
    '''
    A function to run Hamiltonian Monte Carlo for GP regression model

    ----------Inputs---------
    gpr: a pyro GP regression model
    num_samples: number of samples to draw from the posterior
    warmup_steps: number of warmup steps for NUTS
    target_accept_prob: target acceptance probability for NUTS
    print_stats: whether to print the states of the model

    ----------Outputs---------
    mcmc: a pyro MCMC object
    
    '''
    hmc_kernel = HMC(gpr.model,target_accept_prob=target_accept_prob)
    mcmc = MCMC(hmc_kernel, num_samples=num_samples,warmup_steps=warmup_steps)
    mcmc.run()
    if print_stats:
        for name, value in mcmc.get_samples().items():
            if 'kernel' in name:
                if value.dim()==1:
                    print('-----{}: {:4.2f} +/ {:4.2f} (2sd)-----'.format(name,value.mean(),2*value.std()))
                    print('Gelman-Rubin statistic for {}: {:4.2f}'.format(name,mcmc.diagnostics()[name]['r_hat'].item()))
                    print('Effective sample size for {}: {:4.2f}'.format(name,mcmc.diagnostics()[name]['n_eff'].item()))

    return mcmc

def opti_pyro_model(model, X, y, x_sigma, y_sigma, *args, lr=0.05, number_of_steps=500, step_size=100, decay_r=0.1):
    """
    A function to optimize the pyro model

    Parameters:
    model: function, the model function to be optimized
    X: torch.Tensor, the input features
    y: torch.Tensor, the target variable
    x_sigma: float, standard deviation of the input features' error
    y_sigma: float, standard deviation of the target variable's error
    *args: additional arguments for the model
    lr: float, initial learning rate
    number_of_steps: int, total number of optimization steps
    step_size: int, the number of steps after which the learning rate will decay
    decay_r: float, the decay rate by which the learning rate will be multiplied

    Returns:
    guide: the optimized guide corresponding to the model
    losses: list of loss values during training
    """
    # Initialize the guide function
    guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)

    # Setup the optimizer with initial learning rate
    adam = Adam({"lr": lr})
    scheduler = ExponentialLR({'optimizer': adam, 'optim_args': {'lr': step_size}, 'gamma': decay_r})

    # Setup the SVI object for stochastic variational inference
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    # Training loop
    pyro.clear_param_store()
    losses = []
    for step in tqdm(range(number_of_steps)):
        # Perform a step of SVI
        loss = svi.step(X, y, x_sigma, y_sigma, *args)
        losses.append(loss / len(X))
        # Update the learning rate
        if (step + 1) % step_size == 0:
            scheduler.step()

    return guide, losses

def mcmc_pyro_model(model,model_paras,num_samples=2500,warmup_steps=200,target_accept_prob = 0.8,print_stats=False):
    '''
    A function to run MCMC model in Pyro

    ------------Inputs--------------
    model: pyro model
    model_paras: dictionary, the parameters of the model
    num_samples: int, number of samples to draw
    warmup_steps: int, number of warmup steps
    target_accept_prob: float, target acceptance probability
    print_stats: bool, whether to print the statistics of the model

    ------------Outputs--------------
    hmc_samples: dictionary, the samples of the model
    '''
    
    nuts_kernel = NUTS(model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc.run(*model_paras)
    
    if print_stats:
        for name, value in mcmc.get_samples().items():
            if value.dim()==1:
                print('-----{}: {:4.2f} +/ {:4.2f} (2sd)-----'.format(name,value.mean(),2*value.std()))
                print('Gelman-Rubin statistic for {}: {:4.2f}'.format(name,mcmc.diagnostics()[name]['r_hat'].item()))
                print('Effective sample size for {}: {:4.2f}'.format(name,mcmc.diagnostics()[name]['n_eff'].item()))
    return mcmc

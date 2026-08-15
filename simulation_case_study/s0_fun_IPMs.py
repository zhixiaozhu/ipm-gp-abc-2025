import pandas as pd
import numpy as np
from gpflow.models.model import GPModel
import scipy.stats as stats
import pickle
import os
import matplotlib.pylab as plt


####################    IPM for MLEs of GLMs and (uncached) GPs    ##############################################################

# p matrix for MLEs under sep setting.
def P_kernel_fun_sep(mesh, models):
    Xp = mesh[:, None]

    if isinstance(models['m_sur'], GPModel):
        s = np.array(models["m_sur"].predict_y(Xp)[0])
        
        p_f = models["m_grw_f"].predict_y(Xp)
        mean_f = np.array(p_f[0]) 
        var_f = np.array(p_f[1])

        p_nf = models["m_grw_nf"].predict_y(Xp)
        mean_nf = np.array(p_nf[0])
        var_nf = np.array(p_nf[1])

        f = np.array(models["m_fec"].predict_y(Xp)[0])

    else:
        Xp = pd.DataFrame(Xp, columns=['size'])
        s = models["m_sur"].predict(Xp).values[:, None]

        mean_f = models["m_grw_f"].predict(Xp).values[:, None]
        var_f = models["m_grw_f"].mse_resid

        mean_nf = models["m_grw_nf"].predict(Xp).values[:, None]
        var_nf = models["m_grw_nf"].mse_resid

        f = models["m_fec"].predict(Xp).values[:, None]


    g_total_f = stats.norm(mean_f, np.sqrt(var_f)).pdf(mesh)
    g_total_nf = stats.norm(mean_nf, np.sqrt(var_nf)).pdf(mesh)

    return (s*f*g_total_f).T + (s*(1-f)*g_total_nf).T 


# p matrix for MLEs under nonsep setting.
def P_kernel_fun_nonsep(mesh, models):
    # Calculating P matrix under with the same growth kernel for both breeders and non-breeders
    Xp = mesh[:, None]

    if isinstance(models['m_sur'], GPModel):
        s = np.array(models["m_sur"].predict_y(Xp)[0])
        
        p = models["m_grw"].predict_y(Xp)
        mu = np.array(p[0]) 
        sd = np.sqrt(np.array(p[1]))
        g_total = stats.norm(mu, sd).pdf(mesh)

    else:
        Xp = pd.DataFrame(Xp, columns=['size'])
        s = models["m_sur"].predict(Xp).values[:, None]
        
        mu = models["m_grw"].predict(Xp).values[:, None]
        sd = np.sqrt(models["m_grw"].mse_resid)
        g_total = stats.norm(mu, sd).pdf(mesh)
 
    return (s*g_total).T

# 2 - F kernels 
# F matrix for MLEs 
# reproduction occurs first, so there is no survival rate in F kernel
def F_kernel_fun(mesh, models):
    Xp = mesh[:, None]

    if isinstance(models['m_fec'], GPModel): 
        fec = np.array(models["m_fec"].predict_y(Xp)[0])
        flow = np.array(models["m_flow_poi"].predict_y(Xp)[0]+1)
    else:
        Xp = pd.DataFrame(Xp, columns=['size'])
        fec = models["m_fec"].predict(Xp).values[:, None]
        flow = models["m_flow_poi"].predict(Xp).values[:, None]+1

    r_est = models["recruit_p"]    
    alpha = models["alpha"]
    beta = models["beta"]
    r_size = stats.gamma(a=alpha, scale=1/beta).pdf(mesh)

    return (fec*flow*r_est*r_size).T


# a function for fixing evictions by using the "constant" method (for MLEs ).
def constant_fix(models, mesh, P):
    Xp = mesh[:, None]
    if isinstance(models['m_sur'], GPModel): 
        s = np.array(models["m_sur"].predict_y(Xp)[0])
    else: 
        Xp = pd.DataFrame(Xp, columns=['size'])
        s = models["m_sur"].predict(Xp).values[:, None] 
    
    col_sum = P.sum(axis=0)
    P_new = (P.T/col_sum[:, None]).T * s.T
    
    return P_new


def kernel_setting(m_size, lower, upper):
    h = (upper - lower) / m_size
    mesh = (np.arange(m_size) + 0.5)* h + lower
    return([mesh, h, m_size])

# K matrix for MLEs
def IPM_listk(mesh_setting, models, grw_setting):
    mesh = mesh_setting[0]
    h = mesh_setting[1]
    F_matrix = F_kernel_fun(mesh, models) * h
    
    if grw_setting == 'sep':
        P_matrix = P_kernel_fun_sep(mesh, models) * h
    elif grw_setting == 'nonsep': 
        P_matrix = P_kernel_fun_nonsep(mesh, models) * h
    else:
        assert False, '\n grw_setting should be sep or nonsep'

    # constant correction
    P_matrix = constant_fix(models, mesh, P=P_matrix)

    return (P_matrix, F_matrix)


#########################################    IPMs for GLMs (MCMC)   ##########################################################

# p matrix for glm MCMC under sep setting.
def P_kernel_sep_glmmcmc(mesh, models, traces, index):

    if len(index) != 5:
        assert False, '\n index should be with length 5 when considering different growth kernels for reproducers and non-reproducers.' 

    Xp = pd.DataFrame(mesh, columns=['size'])

    s = models["m_sur"].predict(traces["trace_sur"], 
                                data=Xp, inplace=False).posterior['sur_mean'].values[0][index[2], : ][:, None]

    
    p_grw_f = models["m_grw_f"].predict(traces["trace_grw_f"], data=Xp, inplace=False)
    mean_f = p_grw_f.posterior['sizeNext_mean'].values[0][index[3], : ][:, None]
    var_f = p_grw_f.posterior['sizeNext_sigma'].values[0][index[3]] ** 2

    p_grw_nf = models["m_grw_nf"].predict(traces["trace_grw_nf"], data=Xp, inplace=False)
    mean_nf = p_grw_nf.posterior['sizeNext_mean'].values[0][index[4], : ][:, None]
    var_nf = p_grw_nf.posterior['sizeNext_sigma'].values[0][index[4]] ** 2

    f = models["m_fec"].predict(traces["trace_fec"], 
                                data=Xp, inplace=False).posterior['fec_mean'].values[0][index[0], : ][:, None]


    g_total_f = stats.norm(mean_f, np.sqrt(var_f)).pdf(mesh)
    g_total_nf = stats.norm(mean_nf, np.sqrt(var_nf)).pdf(mesh)

    return (s*f*g_total_f).T + (s*(1-f)*g_total_nf).T 
 

# p matrix for glm MCMC under nonsep setting.
def P_kernel_nonsep_glmmcmc(mesh, models, traces, index):

    if len(index) != 4:
        assert False, '\n index should be with length 4 when considering the same growth kernel.' 

    Xp = pd.DataFrame(mesh, columns=['size'])

    s = models["m_sur"].predict(traces["trace_sur"], 
                                data=Xp, inplace=False).posterior['sur_mean'].values[0][index[2], : ][:, None]
 
    p_grw = models["m_grw"].predict(traces["trace_grw"], data=Xp, inplace=False)
    mu = p_grw.posterior['sizeNext_mean'].values[0][index[3], : ][:, None]
    sd = p_grw.posterior['sizeNext_sigma'].values[0][index[3]]

    g_total = stats.norm(mu, sd).pdf(mesh)
 
    return (s*g_total).T



# 2 - F kernels
# F matrix for glm MCMC 
# reproduction occurs first, so there is no survival rate in F kernel
def F_kernel_glmmcmc(mesh, models, traces, index):
    Xp = pd.DataFrame(mesh, columns=['size'])

    fec = models["m_fec"].predict(traces["trace_fec"], 
                                  data=Xp, inplace=False).posterior['fec_mean'].values[0][index[0], : ][:, None]

    flow = models["m_flow_poi"].predict(traces["trace_flow"], 
                                    data=Xp, inplace=False).posterior['flow_mean'].values[0][index[1], : ][:, None] + 1


    r_est = models["recruit_p"]    
    alpha = models["alpha"]
    beta = models["beta"]
    r_size = stats.gamma(a=alpha, scale=1/beta).pdf(mesh)

    return (fec*flow*r_est*r_size).T





# a function for fixing evictions by using the "constant" method (for MLEs ).
def constant_fix_glmmcmc(models, mesh, traces, index, P):
    Xp = pd.DataFrame(mesh, columns=['size'])
    s = models["m_sur"].predict(traces["trace_sur"], 
                                data=Xp, inplace=False).posterior['sur_mean'].values[0][index[2], : ][:, None]
    
    col_sum = P.sum(axis=0)
    P_new = (P.T/col_sum[:, None]).T * s.T
    
    return P_new

# K matrix for GLM MCMC
def IPM_listk_glmmcmc(mesh_setting, models, traces, index, grw_setting):
    mesh = mesh_setting[0]
    h = mesh_setting[1]
    
    if grw_setting == 'sep':
        P_matrix = P_kernel_sep_glmmcmc(mesh=mesh, models=models, traces=traces, index=index) * h
    elif grw_setting == 'nonsep' : 
        P_matrix = P_kernel_nonsep_glmmcmc(mesh=mesh, models=models, traces=traces, index=index) * h
    else:
        assert False, '\n grw_setting should be sep or nonsep'
    F_matrix = F_kernel_glmmcmc(mesh=mesh, models=models, traces=traces, index=index) * h
    # constant correction

    P_matrix = constant_fix_glmmcmc(mesh=mesh, models=models, traces=traces, index=index, P=P_matrix)

    return (P_matrix, F_matrix)




####################    IPM for cached GP MCMCs    #####################################################
from s0_fun_IBMs import GPMC_posterior, predict_y_loaded_cache, predict_f_loaded_cache

# p matrix for MLEs under sep setting.
def P_kernel_fun_sepcache(mesh, models):
    Xp = mesh[:, None]

    s = np.array(predict_y_loaded_cache(model=models["m_sur"], Xnew=Xp, Cache=models["m_sur"].cache)[0])
    
    p_f = predict_y_loaded_cache(model=models["m_grw_f"], Xnew=Xp, Cache=models["m_grw_f"].cache) 
    mean_f = np.array(p_f[0]) 
    var_f = np.array(p_f[1])

    p_nf = predict_y_loaded_cache(model=models["m_grw_nf"], Xnew=Xp, Cache=models["m_grw_nf"].cache) 
    mean_nf = np.array(p_nf[0])
    var_nf = np.array(p_nf[1])

    f = np.array(predict_y_loaded_cache(model=models["m_fec"], Xnew=Xp, Cache=models["m_fec"].cache)[0])

    g_total_f = stats.norm(mean_f, np.sqrt(var_f)).pdf(mesh)
    g_total_nf = stats.norm(mean_nf, np.sqrt(var_nf)).pdf(mesh)

    return (s*f*g_total_f).T + (s*(1-f)*g_total_nf).T 


# p matrix for MLEs under nonsep setting.
def P_kernel_fun_nonsepcache(mesh, models):
    # Calculating P matrix under with the same growth kernel for both breeders and non-breeders
    Xp = mesh[:, None]

    s = np.array(predict_y_loaded_cache(model=models["m_sur"], Xnew=Xp, Cache=models["m_sur"].cache)[0])
    p = predict_y_loaded_cache(model=models["m_grw"], Xnew=Xp, Cache=models["m_grw"].cache) 
    mu = np.array(p[0]) 
    sd = np.sqrt(np.array(p[1]))
    g_total = stats.norm(mu, sd).pdf(mesh)

    return (s*g_total).T


# 2 - F kernels 
# F matrix for MLEs 
# reproduction occurs first, so there is no survival rate in F kernel
def F_kernel_funcache(mesh, models):
    Xp = mesh[:, None]

    fec = np.array(predict_y_loaded_cache(model=models["m_fec"], Xnew=Xp, Cache=models["m_fec"].cache)[0]) 
    flow = np.array(predict_y_loaded_cache(model=models["m_flow_poi"], Xnew=Xp, Cache=models["m_flow_poi"].cache)[0]+1)

    r_est = models["recruit_p"]    
    alpha = models["alpha"]
    beta = models["beta"]
    r_size = stats.gamma(a=alpha, scale=1/beta).pdf(mesh)

    return (fec*flow*r_est*r_size).T


# a function for fixing evictions by using the "constant" method (for MLEs ).
def constant_fixcache(models, mesh, P):
    Xp = mesh[:, None]

    s = np.array(predict_y_loaded_cache(model=models["m_sur"], Xnew=Xp, Cache=models["m_sur"].cache)[0])
        
    col_sum = P.sum(axis=0)
    P_new = (P.T/col_sum[:, None]).T * s.T
    
    return P_new


# K matrix for MLEs
def IPM_listk_gpcache(mesh_setting, models, grw_setting):
    mesh = mesh_setting[0]
    h = mesh_setting[1]
    F_matrix = F_kernel_funcache(mesh, models) * h
    
    if grw_setting == 'sep':
        P_matrix = P_kernel_fun_sepcache(mesh, models) * h
    elif grw_setting == 'nonsep': 
        P_matrix = P_kernel_fun_nonsepcache(mesh, models) * h
    else:
        assert False, '\n grw_setting should be sep or nonsep'
    # constant correction
    P_matrix = constant_fixcache(models, mesh, P=P_matrix)

    return (P_matrix, F_matrix)






















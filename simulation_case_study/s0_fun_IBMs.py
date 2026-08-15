# This file contains IBM simulations for different situations.
# Ages will be simulated during IBMs, but GLM and GP models used for predictions do not consider 'age' as a covariate.

#  We are requiring 6 different types of IBMs. 
#   (1) The IBM for MCMC based GLMs when setting DIFFERENT Growth kernel for reproducers
#       and non-reproducers, which is used to simulate populations afterwards.
#   (2) The IBM for MLE based GLMs when setting DIFFERENT Growth kernel for reproducers
#       and non-reproducers, which is used to generate 'true' populations.
#   (3) The IBM for GPMs when setting DIFFERENT Growth kernel for reproducers
#       and non-reproducers, which is used to generate 'true' populations and implement ABC-PMC.
#   (4) The IBM for GLMs when setting the SAME Growth kernel for reproducers
#       and non-reproducers, which is used to implement ABC-PMC.
#   (5) The IBM for GPs when setting the SAME Growth kernel for reproducers
#       and non-reproducers, which is used to implement ABC-PMC.
#   (6) The cached version of (3) make predictions based on the GP models with cached informations.
#   (7) The cached version of (5).

import numpy as np
import pandas as pd
import warnings
import scipy.stats as stats
import bambi as bmb

# (1) The IBM for GLMs when setting DIFFERENT Growth kernel 

def IBM_1step_glm(zt, age, models, trace, index):
    # models: a dictionary containing all GLMs
    # trace: a dictionary containing all kept MCMC traces for each of GLMs.
    # index: a list of index of MCMC samples for each of GLMs, that indicates which models we are going to use.

    if len(index) != 5:
        assert False, '\n index should be with length 5 when considering different growth kernels for reproducers and non-reproducers.' 

    # initialize 
    current_n = np.shape(zt)[0]
    rep_stalks = np.repeat(np.nan, current_n)[:, None]
    surv = np.repeat(np.nan, current_n)[:, None]
    zprime = np.repeat(np.nan, current_n)[:, None]
    ageprime = np.repeat(np.nan, current_n)[:, None]
    num_recruits = 0 
    zt = np.array(zt)[:, None]
    age = np.array(age)[:, None]
    X_t = zt.copy() 

    # we simulate those breeders first
    p_breeding = models['m_fec'].predict(trace["trace_fec"], data=pd.DataFrame(X_t, columns=['size']), inplace=False)
    rep_breeding = np.random.binomial(n = 1, p = p_breeding.posterior['fec_mean'].values[0][index[0], : ])
    whether_bre = (rep_breeding == 1).reshape(1, current_n)[0]        
    num_bre = np.sum(whether_bre)
    num_nonbre = current_n - num_bre


    lambda_t = models["m_flow_poi"].predict(trace["trace_flow"], data=pd.DataFrame(X_t[whether_bre, :], columns=['size']), inplace=False)
    lambda_t = lambda_t.posterior['flow_mean'].values[0][index[1], : ] 
    if np.any(np.isnan(lambda_t)):
        warnings.warn('nan lambda_t produced!!')
        print('nan lambda_t produced!!', flush=True)
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)

    if np.any(np.array(lambda_t) >= 10000):
        warnings.warn('lambda_t is too large !!')
        print('lambda_t is too large !!', flush=True) 
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)
    else:
        rep_stalks[whether_bre] = np.random.poisson(lambda_t)[:, None] + 1    
        
         
    # simulate recruits based on the number of flowering stalks
    if (num_bre != 0):
        num_recruits = np.random.binomial(n = np.nansum(rep_stalks), p = models["recruit_p"])
        # assign size for those recruits
        rep_size = stats.gamma(a = models["alpha"], scale = 1/models["beta"]).rvs(size=num_recruits)[:, None]
            

    # now, simulating surviving.
    p_surv = models['m_sur'].predict(trace["trace_sur"], data=pd.DataFrame(X_t, columns=['size']), inplace=False)
    surv = np.random.binomial(n = 1, p = p_surv.posterior['sur_mean'].values[0][index[2], : ])
    whether_surv = (surv == 1).reshape(1, current_n)[0]
    num_surv = np.sum(whether_surv)

    # let these survivors grow up
    # for breeders
    mean_zprime_f = models["m_grw_f"].predict(trace["trace_grw_f"], data=pd.DataFrame(X_t[whether_surv & whether_bre], columns=['size']), inplace=False)
    zprime[whether_surv & whether_bre] = np.random.normal(mean_zprime_f.posterior['sizeNext_mean'].values[0][index[3], : ], 
                                                          mean_zprime_f.posterior['sizeNext_sigma'].values[0][index[3]])[:, None]

    # for non-breeders
    not_bre = whether_bre == False
    mean_zprime_nf = models["m_grw_nf"].predict(trace["trace_grw_nf"], data=pd.DataFrame(X_t[whether_surv & not_bre], columns=['size']), inplace=False)
    zprime[whether_surv & not_bre] = np.random.normal(mean_zprime_nf.posterior['sizeNext_mean'].values[0][index[4], : ], 
                                                      mean_zprime_nf.posterior['sizeNext_sigma'].values[0][index[4]])[:, None]
    
    # in our case, age 13 is an absorbing state.
    ageprime[whether_surv] = np.minimum(age[whether_surv]+1, 13)    

    rep_breeding = rep_breeding[:, None]
    surv = surv[:, None]
    # store the simulation data 
    # Here: age is for the age at current not ageNext
    if (num_bre != 0):
        zprime = np.concatenate((zprime, rep_size))
        zt = np.concatenate((zt, np.repeat(np.nan, num_recruits)[:, None]))
        rep_breeding = np.concatenate((rep_breeding, np.repeat(np.nan, num_recruits)[:, None]))
        rep_stalks = np.concatenate((rep_stalks, np.repeat(np.nan, num_recruits)[:, None]))
        surv = np.concatenate((surv, np.repeat(np.nan, num_recruits)[:, None]))
        age = np.concatenate((age, np.repeat(np.nan, num_recruits)[:, None]))
        ageprime = np.concatenate((ageprime, np.repeat(1, num_recruits)[:, None]))

    data = pd.DataFrame(np.concatenate((zt, zprime, rep_breeding, rep_stalks, surv, age, ageprime), axis=1),
                        columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

    return(data)


# (2) The IBM for MLE based GLMs when setting DIFFERENT Growth kernel 

def IBM_1step_glm_mle(zt, age, models):
    # models: a dictionary containing all GLMs
    # trace: a dictionary containing all kept MCMC traces for each of GLMs.
    # index: a list of index of MCMC samples for each of GLMs, that indicates which models we are going to use.
 
    # initialize 
    current_n = np.shape(zt)[0]
    rep_stalks = np.repeat(np.nan, current_n)[:, None]
    surv = np.repeat(np.nan, current_n)[:, None]
    zprime = np.repeat(np.nan, current_n)[:, None]
    ageprime = np.repeat(np.nan, current_n)[:, None]
    num_recruits = 0 
    zt = np.array(zt)[:, None]
    age = np.array(age)[:, None]
    X_t = zt.copy() 

    # we simulate those breeders first
    p_breeding = models['m_fec'].predict(pd.DataFrame(X_t, columns=['size'])).values[:, None]
    rep_breeding = np.random.binomial(n = 1, p = p_breeding)
    whether_bre = (rep_breeding == 1).reshape(1, current_n)[0]        
    num_bre = np.sum(whether_bre)
    num_nonbre = current_n - num_bre


    lambda_t = models["m_flow_poi"].predict(pd.DataFrame(X_t[whether_bre, :], columns=['size'])).values[:, None]
    if np.any(np.isnan(lambda_t)):
        warnings.warn('nan lambda_t produced!!')
        print('nan lambda_t produced!!', flush=True)
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)

    if np.any(np.array(lambda_t) >= 10000):
        warnings.warn('lambda_t is too large !!')
        print('lambda_t is too large !!', flush=True) 
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)
    else:
        rep_stalks[whether_bre] = np.random.poisson(lambda_t) + 1    
        
         
    # simulate recruits based on the number of flowering stalks
    if (num_bre != 0):
        num_recruits = np.random.binomial(n = np.nansum(rep_stalks), p = models["recruit_p"])
        # assign size for those recruits
        rep_size = stats.gamma(a = models["alpha"], scale = 1/models["beta"]).rvs(size=num_recruits)[:, None]
            

    # now, simulating surviving.
    p_surv = models['m_sur'].predict(pd.DataFrame(X_t, columns=['size'])).values[:, None]
    surv = np.random.binomial(n = 1, p = p_surv)
    whether_surv = (surv == 1).reshape(1, current_n)[0]
    num_surv = np.sum(whether_surv)

    # let these survivors grow up
    # for breeders
    X_t_f = pd.DataFrame(X_t[whether_surv & whether_bre], columns=['size'])
    mean_zprime_f = models["m_grw_f"].predict(X_t_f).values[:, None]
    # here, to make predictions, we should use std for predictions instead of that for the expected values.
    zprime[whether_surv & whether_bre] = np.random.normal(mean_zprime_f, np.sqrt(models["m_grw_f"].mse_resid))

    # for non-breeders
    not_bre = whether_bre == False
    X_t_nf = pd.DataFrame(X_t[whether_surv & not_bre], columns=['size'])
    mean_zprime_nf = models["m_grw_nf"].predict(X_t_nf).values[:, None]
    zprime[whether_surv & not_bre] = np.random.normal(mean_zprime_nf, np.sqrt(models["m_grw_nf"].mse_resid))
    
    # in our case, age 13 is an absorbing state.
    ageprime[whether_surv] = np.minimum(age[whether_surv]+1, 13)    


    # store the simulation data 
    # Here: age is for the age at current not ageNext
    if (num_bre != 0):
        zprime = np.concatenate((zprime, rep_size))
        # zprime = np.minimum(zprime, 5)    
        zt = np.concatenate((zt, np.repeat(np.nan, num_recruits)[:, None]))
        rep_breeding = np.concatenate((rep_breeding, np.repeat(np.nan, num_recruits)[:, None]))
        rep_stalks = np.concatenate((rep_stalks, np.repeat(np.nan, num_recruits)[:, None]))
        surv = np.concatenate((surv, np.repeat(np.nan, num_recruits)[:, None]))
        age = np.concatenate((age, np.repeat(np.nan, num_recruits)[:, None]))
        ageprime = np.concatenate((ageprime, np.repeat(1, num_recruits)[:, None]))

    data = pd.DataFrame(np.concatenate((zt, zprime, rep_breeding, rep_stalks, surv, age, ageprime), axis=1),
                        columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

    return(data)




# (3) The IBM for GPs when setting DIFFERENT Growth kernel 

def IBM_1step_gp(zt, age, models):
    # initialize 

    current_n = np.shape(zt)[0]
    rep_stalks = np.repeat(np.nan, current_n)[:, None]
    surv = np.repeat(np.nan, current_n)[:, None]
    zprime = np.repeat(np.nan, current_n)[:, None]
    ageprime = np.repeat(np.nan, current_n)[:, None]
    num_recruits = 0 
    zt = np.array(zt)[:, None]
    age = np.array(age)[:, None]
    X_t = zt.copy() 

    # we simulate those breeders first
    p_breeding = models["m_fec"].predict_y(X_t)[0]
    rep_breeding = np.random.binomial(n = 1, p = p_breeding)
    whether_bre = (rep_breeding == 1).reshape(1, current_n)[0]        
    num_bre = np.sum(whether_bre)
    num_nonbre = current_n - num_bre

    lambda_t = models["m_flow_poi"].predict_y(X_t[whether_bre, :])[0]
    if np.any(np.isnan(lambda_t)):
        warnings.warn('nan lambda_t produced!!')
        print('nan lambda_t produced!!', flush=True)
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)

    if np.any(np.array(lambda_t) >= 10000):
        warnings.warn('lambda_t is too large !!')
        print('lambda_t is too large !!', flush=True) 
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)
    else:
        rep_stalks[whether_bre] = np.random.poisson(lambda_t) + 1
           
    # simulate recruits based on the number of flowering stalks
    if (num_bre != 0):
        num_recruits = np.random.binomial(n = np.nansum(rep_stalks), p = models["recruit_p"])
        # assign size for those recruits
        rep_size = stats.gamma(a = models["alpha"], scale = 1/models["beta"]).rvs(size=num_recruits)[:, None]
            
        
    # now, simulating surviving.
    p_surv = models["m_sur"].predict_y(X_t)[0]
    surv = np.random.binomial(n = 1, p = p_surv)
    whether_surv = (surv == 1).reshape(1, current_n)[0]
    num_surv = np.sum(whether_surv)

    # let these survivors grow up
    # for breeders
    X_t_f = X_t[whether_surv & whether_bre]
    mean_zprime_f = models["m_grw_f"].predict_y(X_t_f)
    zprime[whether_surv & whether_bre] = np.random.normal(mean_zprime_f[0], 
                                                          np.sqrt(mean_zprime_f[1]))
    
    # for non-breeders
    not_bre = whether_bre == False
    X_t_nf = X_t[whether_surv & not_bre]
    mean_zprime_nf = models["m_grw_nf"].predict_y(X_t_nf)
    zprime[whether_surv & not_bre] = np.random.normal(mean_zprime_nf[0], 
                                                      np.sqrt(mean_zprime_nf[1]))
    
    # in our case, age 13 is an absorbing state.
    ageprime[whether_surv] = np.minimum(age[whether_surv]+1, 13)
    
    # store the simulation data 
    # Here: age is for the age at current not ageNext
    if (num_bre != 0):
        zprime = np.concatenate((zprime, rep_size))
        zt = np.concatenate((zt, np.repeat(np.nan, num_recruits)[:, None]))
        rep_breeding = np.concatenate((rep_breeding, np.repeat(np.nan, num_recruits)[:, None]))
        rep_stalks = np.concatenate((rep_stalks, np.repeat(np.nan, num_recruits)[:, None]))
        surv = np.concatenate((surv, np.repeat(np.nan, num_recruits)[:, None]))
        age = np.concatenate((age, np.repeat(np.nan, num_recruits)[:, None]))
        ageprime = np.concatenate((ageprime, np.repeat(1, num_recruits)[:, None]))

    data = pd.DataFrame(np.concatenate((zt, zprime, rep_breeding, rep_stalks, surv, age, ageprime), axis=1),
                        columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

    return(data)



# (4) The IBM for GLMs when setting the SAME Growth kernel 

def IBM_1step_glm_same(zt, age, models, trace, index):
    # models: a dictionary containing all GLMs
    # trace: a dictionary containing all kept MCMC traces for each of GLMs.
    # index: a list of index of MCMC samples for each of GLMs, that indicates which models we are going to use.

    if len(index) != 4:
        assert False, '\n index should be with length 4 when considering the same growth kernel.'

    # initialize 
    current_n = np.shape(zt)[0]
    rep_stalks = np.repeat(np.nan, current_n)[:, None]
    surv = np.repeat(np.nan, current_n)[:, None]
    zprime = np.repeat(np.nan, current_n)[:, None]
    ageprime = np.repeat(np.nan, current_n)[:, None]
    num_recruits = 0 
    zt = np.array(zt)[:, None]
    age = np.array(age)[:, None]
    X_t = zt.copy() 

    # we simulate those breeders first
    p_breeding = models['m_fec'].predict(trace["trace_fec"], data=pd.DataFrame(X_t, columns=['size']), inplace=False)
    rep_breeding = np.random.binomial(n = 1, p = p_breeding.posterior['fec_mean'].values[0][index[0], : ])
    whether_bre = (rep_breeding == 1).reshape(1, current_n)[0]        
    num_bre = np.sum(whether_bre)
    num_nonbre = current_n - num_bre


    lambda_t = models["m_flow_poi"].predict(trace["trace_flow"], data=pd.DataFrame(X_t[whether_bre, :], columns=['size']), inplace=False)
    lambda_t = lambda_t.posterior['flow_mean'].values[0][index[1], : ] 
    if np.any(np.isnan(lambda_t)):
        warnings.warn('nan lambda_t produced!!')
        print('nan lambda_t produced!!', flush=True)
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)

    if np.any(np.array(lambda_t) >= 10000):
        warnings.warn('lambda_t is too large !!')
        print('lambda_t is too large !!', flush=True) 
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)
    else:
        rep_stalks[whether_bre] = np.random.poisson(lambda_t)[:, None] + 1    
        
         
    # simulate recruits based on the number of flowering stalks
    if (num_bre != 0):
        num_recruits = np.random.binomial(n = np.nansum(rep_stalks), p = models["recruit_p"])
        # assign size for those recruits
        rep_size = stats.gamma(a = models["alpha"], scale = 1/models["beta"]).rvs(size=num_recruits)[:, None]
            

    # now, simulating surviving.
    p_surv = models['m_sur'].predict(trace["trace_sur"], data=pd.DataFrame(X_t, columns=['size']), inplace=False)
    surv = np.random.binomial(n = 1, p = p_surv.posterior['sur_mean'].values[0][index[2], : ])
    whether_surv = (surv == 1).reshape(1, current_n)[0]
    num_surv = np.sum(whether_surv)

    # let these survivors grow up
    # we are considering the Same groth kernel for breeders and non-breeders.
    mean_zprime = models["m_grw"].predict(trace["trace_grw"], data=pd.DataFrame(X_t[whether_surv], columns=['size']), inplace=False)
    zprime[whether_surv] = np.random.normal(mean_zprime.posterior['sizeNext_mean'].values[0][index[3], : ], 
                                            mean_zprime.posterior['sizeNext_sigma'].values[0][index[3]])[:, None]

    # in our case, age 13 is an absorbing state.
    ageprime[whether_surv] = np.minimum(age[whether_surv]+1, 13)    

    rep_breeding = rep_breeding[:, None]
    surv = surv[:, None]
    # store the simulation data 
    # Here: age is for the age at current not ageNext
    if (num_bre != 0):
        zprime = np.concatenate((zprime, rep_size))
        zt = np.concatenate((zt, np.repeat(np.nan, num_recruits)[:, None]))
        rep_breeding = np.concatenate((rep_breeding, np.repeat(np.nan, num_recruits)[:, None]))
        rep_stalks = np.concatenate((rep_stalks, np.repeat(np.nan, num_recruits)[:, None]))
        surv = np.concatenate((surv, np.repeat(np.nan, num_recruits)[:, None]))
        age = np.concatenate((age, np.repeat(np.nan, num_recruits)[:, None]))
        ageprime = np.concatenate((ageprime, np.repeat(1, num_recruits)[:, None]))

    data = pd.DataFrame(np.concatenate((zt, zprime, rep_breeding, rep_stalks, surv, age, ageprime), axis=1),
                        columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

    return(data)




# (5) The IBM for GPs when setting the SAME Growth kernel 
def IBM_1step_gp_same(zt, age, models):
    # initialize 
    current_n = np.shape(zt)[0]
    rep_stalks = np.repeat(np.nan, current_n)[:, None]
    surv = np.repeat(np.nan, current_n)[:, None]
    zprime = np.repeat(np.nan, current_n)[:, None]
    ageprime = np.repeat(np.nan, current_n)[:, None]
    num_recruits = 0 
    zt = np.array(zt)[:, None]
    age = np.array(age)[:, None]
    X_t = zt.copy()

    # we simulate those breeders first
    p_breeding = models["m_fec"].predict_y(X_t)[0]
    rep_breeding = np.random.binomial(n = 1, p = p_breeding)
    whether_bre = (rep_breeding == 1).reshape(1, current_n)[0]        
    num_bre = np.sum(whether_bre)
    num_nonbre = current_n - num_bre

    lambda_t = models["m_flow_poi"].predict_y(X_t[whether_bre, :])[0]
    if np.any(np.isnan(lambda_t)):
        warnings.warn('nan lambda_t produced!!')
        print('nan lambda_t produced!!', flush=True)
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)

    if np.any(np.array(lambda_t) >= 10000):
        warnings.warn('lambda_t is too large !!')
        print('lambda_t is too large !!', flush=True) 
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)
    else:
        rep_stalks[whether_bre] = np.random.poisson(lambda_t) + 1
           
    # simulate recruits based on the number of flowering stalks
    if (num_bre != 0):
        num_recruits = np.random.binomial(n = np.nansum(rep_stalks), p = models["recruit_p"])
        # assign size for those recruits
        rep_size = stats.gamma(a = models["alpha"], scale = 1/models["beta"]).rvs(size=num_recruits)[:, None]
            
        
    # now, simulating surviving.
    p_surv = models["m_sur"].predict_y(X_t)[0]
    surv = np.random.binomial(n = 1, p = p_surv)
    whether_surv = (surv == 1).reshape(1, current_n)[0]
    num_surv = np.sum(whether_surv)

    # let these survivors grow up
    # we are considering the Same groth kernel for breeders and non-breeders.
    mean_zprime = models["m_grw"].predict_y(X_t[whether_surv])
    zprime[whether_surv] = np.random.normal(mean_zprime[0], np.sqrt(mean_zprime[1]))

    # in our case, age 13 is an absorbing state.
    ageprime[whether_surv] = np.minimum(age[whether_surv]+1, 13)
    
    # store the simulation data 
    # Here: age is for the age at current not ageNext
    if (num_bre != 0):
        zprime = np.concatenate((zprime, rep_size))
        zt = np.concatenate((zt, np.repeat(np.nan, num_recruits)[:, None]))
        rep_breeding = np.concatenate((rep_breeding, np.repeat(np.nan, num_recruits)[:, None]))
        rep_stalks = np.concatenate((rep_stalks, np.repeat(np.nan, num_recruits)[:, None]))
        surv = np.concatenate((surv, np.repeat(np.nan, num_recruits)[:, None]))
        age = np.concatenate((age, np.repeat(np.nan, num_recruits)[:, None]))
        ageprime = np.concatenate((ageprime, np.repeat(1, num_recruits)[:, None]))

    data = pd.DataFrame(np.concatenate((zt, zprime, rep_breeding, rep_stalks, surv, age, ageprime), axis=1),
                        columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

    return(data)





# Functions used to connect gpflow functions with the cached information are written in the end of this script.


# (6) cached version of (3) 
def IBM_1step_gp_cache(zt, age, models):

    # initialize 
    current_n = np.shape(zt)[0]
    rep_stalks = np.repeat(np.nan, current_n)[:, None]
    surv = np.repeat(np.nan, current_n)[:, None]
    zprime = np.repeat(np.nan, current_n)[:, None]
    ageprime = np.repeat(np.nan, current_n)[:, None]
    num_recruits = 0 
    zt = np.array(zt)[:, None]
    age = np.array(age)[:, None]
    X_t = zt.copy() 

    # we simulate those breeders first
    p_breeding = predict_y_loaded_cache(model=models["m_fec"], Xnew=X_t, Cache=models["m_fec"].cache)[0]
    rep_breeding = np.random.binomial(n = 1, p = p_breeding)
    whether_bre = (rep_breeding == 1).reshape(1, current_n)[0]        
    num_bre = np.sum(whether_bre)
    num_nonbre = current_n - num_bre

    lambda_t = predict_y_loaded_cache(model=models["m_flow_poi"], Xnew=X_t[whether_bre, :], Cache=models["m_flow_poi"].cache)[0]
    if np.any(np.isnan(lambda_t)):
        warnings.warn('nan lambda_t produced!!')
        print('nan lambda_t produced!!', flush=True)
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)

    if np.any(np.array(lambda_t) >= 10000):
        warnings.warn('lambda_t is too large !!')
        print('lambda_t is too large !!', flush=True) 
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)
    else:
        rep_stalks[whether_bre] = np.random.poisson(lambda_t) + 1
           
    # simulate recruits based on the number of flowering stalks
    if (num_bre != 0):
        num_recruits = np.random.binomial(n = np.nansum(rep_stalks), p = models["recruit_p"])
        # assign size for those recruits
        rep_size = stats.gamma(a = models["alpha"], scale = 1/models["beta"]).rvs(size=num_recruits)[:, None]
            
        
    # now, simulating surviving.
    p_surv = predict_y_loaded_cache(model=models["m_sur"], Xnew=X_t, Cache=models["m_sur"].cache)[0]
    surv = np.random.binomial(n = 1, p = p_surv)
    whether_surv = (surv == 1).reshape(1, current_n)[0]
    num_surv = np.sum(whether_surv)

    # let these survivors grow up
    # for breeders
    X_t_f = X_t[whether_surv & whether_bre]
    mean_zprime_f = predict_y_loaded_cache(model=models["m_grw_f"], Xnew=X_t_f, Cache=models["m_grw_f"].cache) 
    zprime[whether_surv & whether_bre] = np.random.normal(mean_zprime_f[0], 
                                                          np.sqrt(mean_zprime_f[1]))
    
    # for non-breeders
    not_bre = whether_bre == False
    X_t_nf = X_t[whether_surv & not_bre]
    mean_zprime_nf = predict_y_loaded_cache(model=models["m_grw_nf"], Xnew=X_t_nf, Cache=models["m_grw_nf"].cache)
    zprime[whether_surv & not_bre] = np.random.normal(mean_zprime_nf[0], 
                                                      np.sqrt(mean_zprime_nf[1]))
    
    # in our case, age 13 is an absorbing state.
    ageprime[whether_surv] = np.minimum(age[whether_surv]+1, 13)
    
    # store the simulation data 
    # Here: age is for the gae at current not ageNext
    if (num_bre != 0):
        zprime = np.concatenate((zprime, rep_size))
        zt = np.concatenate((zt, np.repeat(np.nan, num_recruits)[:, None]))
        rep_breeding = np.concatenate((rep_breeding, np.repeat(np.nan, num_recruits)[:, None]))
        rep_stalks = np.concatenate((rep_stalks, np.repeat(np.nan, num_recruits)[:, None]))
        surv = np.concatenate((surv, np.repeat(np.nan, num_recruits)[:, None]))
        age = np.concatenate((age, np.repeat(np.nan, num_recruits)[:, None]))
        ageprime = np.concatenate((ageprime, np.repeat(1, num_recruits)[:, None]))

    data = pd.DataFrame(np.concatenate((zt, zprime, rep_breeding, rep_stalks, surv, age, ageprime), axis=1),
                        columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

    return(data)



# (7) cached version of (5) 
def IBM_1step_gp_same_cache(zt, age, models):

    # initialize 
    current_n = np.shape(zt)[0]
    rep_stalks = np.repeat(np.nan, current_n)[:, None]
    surv = np.repeat(np.nan, current_n)[:, None]
    zprime = np.repeat(np.nan, current_n)[:, None]
    ageprime = np.repeat(np.nan, current_n)[:, None]
    num_recruits = 0 
    zt = np.array(zt)[:, None]
    age = np.array(age)[:, None]
    X_t = zt.copy() 

    # we simulate those breeders first
    p_breeding = predict_y_loaded_cache(model=models["m_fec"], Xnew=X_t, Cache=models["m_fec"].cache)[0]
    rep_breeding = np.random.binomial(n = 1, p = p_breeding)
    whether_bre = (rep_breeding == 1).reshape(1, current_n)[0]        
    num_bre = np.sum(whether_bre)
    num_nonbre = current_n - num_bre

    lambda_t = predict_y_loaded_cache(model=models["m_flow_poi"], Xnew=X_t[whether_bre, :], Cache=models["m_flow_poi"].cache)[0]
    if np.any(np.isnan(lambda_t)):
        warnings.warn('nan lambda_t produced!!')
        print('nan lambda_t produced!!', flush=True)
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)

    if np.any(np.array(lambda_t) >= 10000):
        warnings.warn('lambda_t is too large !!')
        print('lambda_t is too large !!', flush=True) 
        data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

        return(data)
    else:
        rep_stalks[whether_bre] = np.random.poisson(lambda_t) + 1
           
    # simulate recruits based on the number of flowering stalks
    if (num_bre != 0):
        num_recruits = np.random.binomial(n = np.nansum(rep_stalks), p = models["recruit_p"])
        # assign size for those recruits
        rep_size = stats.gamma(a = models["alpha"], scale = 1/models["beta"]).rvs(size=num_recruits)[:, None]
            
        
    # now, simulating surviving.
    p_surv = predict_y_loaded_cache(model=models["m_sur"], Xnew=X_t, Cache=models["m_sur"].cache)[0]
    surv = np.random.binomial(n = 1, p = p_surv)
    whether_surv = (surv == 1).reshape(1, current_n)[0]
    num_surv = np.sum(whether_surv)

    # let these survivors grow up
    # we are considering the Same growth kernel for breeders and non-breeders.
    mean_zprime = predict_y_loaded_cache(model=models["m_grw"], Xnew=X_t[whether_surv], Cache=models["m_grw"].cache) 
    zprime[whether_surv] = np.random.normal(mean_zprime[0], np.sqrt(mean_zprime[1]))
    
    # in our case, age 13 is an absorbing state.
    ageprime[whether_surv] = np.minimum(age[whether_surv]+1, 13)
    
    # store the simulation data 
    # Here: age is for the gae at current not ageNext
    if (num_bre != 0):
        zprime = np.concatenate((zprime, rep_size))
        zt = np.concatenate((zt, np.repeat(np.nan, num_recruits)[:, None]))
        rep_breeding = np.concatenate((rep_breeding, np.repeat(np.nan, num_recruits)[:, None]))
        rep_stalks = np.concatenate((rep_stalks, np.repeat(np.nan, num_recruits)[:, None]))
        surv = np.concatenate((surv, np.repeat(np.nan, num_recruits)[:, None]))
        age = np.concatenate((age, np.repeat(np.nan, num_recruits)[:, None]))
        ageprime = np.concatenate((ageprime, np.repeat(1, num_recruits)[:, None]))

    data = pd.DataFrame(np.concatenate((zt, zprime, rep_breeding, rep_stalks, surv, age, ageprime), axis=1),
                        columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

    return(data)





# Cache functions
from typing import Optional, Tuple

import tensorflow as tf
from check_shapes import check_shapes

import gpflow

from gpflow import posteriors
from gpflow.base import InputData, MeanAndVariance
from gpflow.posteriors import VGPPosterior
from gpflow.models.model import GPModel
from gpflow.models.gpr import GPR_deprecated

def predict_f_loaded_cache(
    model: GPModel, 
    Xnew: InputData,
    Cache: Optional[Tuple[tf.Tensor, ...]],
    full_cov: bool = False,
    full_output_cov: bool = False,
) -> MeanAndVariance:
    """
    For backwards compatibility, GPR's predict_f uses the fused (no-cache)
    computation, which is more efficient during training.

    For faster (cached) prediction, predict directly from the posterior object, i.e.,:
        model.posterior().predict_f(Xnew, ...)
    """

    if isinstance(model, gpflow.models.gpr.GPR):
        posterior = model.posterior(posteriors.PrecomputeCacheType.NOCACHE)
        posterior.cache = Cache
        return posterior.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
    
    elif isinstance(model, gpflow.models.gpmc.GPMC):
        X_data, _Y_data = model.data
        posterior = VGPPosterior(
            kernel = model.kernel,
            X = X_data,
            q_mu = model.V,
            q_sqrt = None,
            white = True,
            precompute_cache=None,
        )
        posterior.cache = Cache
        return posterior.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        
    else:
        raise ValueError(f"{model} is not a supported GPmodel type for faster predictions.")
         


def predict_y_loaded_cache(
    model: GPModel, 
    Xnew: InputData,
    Cache: Optional[Tuple[tf.Tensor, ...]],
    full_cov: bool = False,
    full_output_cov: bool = False,
) -> MeanAndVariance:
    """
    For backwards compatibility, GPR's predict_f uses the fused (no-cache)
    computation, which is more efficient during training.

    For faster (cached) prediction, predict directly from the posterior object, i.e.,:
        model.posterior().predict_f(Xnew, ...)
    """

    f_mean, f_var = predict_f_loaded_cache(
        model=model, Xnew=Xnew, Cache=Cache, full_cov=full_cov, full_output_cov=full_output_cov
    )

    return model.likelihood.predict_mean_and_var(Xnew, f_mean, f_var)



def GPMC_posterior(
    model: gpflow.models.gpmc.GPMC,
    precompute_cache: posteriors.PrecomputeCacheType = posteriors.PrecomputeCacheType.TENSOR,
) -> posteriors.VGPPosterior:
    
    X_data, _Y_data = model.data
    return posteriors.VGPPosterior(
        kernel=model.kernel,
        X=X_data,
        q_mu=model.V,
        q_sqrt=None,
        white=True,
        precompute_cache=precompute_cache,
    )



# given dataset at t-1, popu_structure would return the starting population structure at time t
def popu_structure(data):
    survivor_index = data["surv"] == 1
    new_born_index = np.isnan(data["size"])
    new = data.loc[survivor_index | new_born_index]
    return (new.sizeNext, new.ageNext)




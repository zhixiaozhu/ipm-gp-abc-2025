from s0_initial import *


def IBM_full(n0, popu_max, models, total_time, NB=True, year=False, current_year=np.nan):
    # If year=True, this means that the models contains year as another covariate
    if (year==True) and (np.isnan(current_year)):
        assert False, '\n Please indicate the current_year when considering considering year.'


    z0 = stats.gamma(a = models["alpha"], scale = 1/models["beta"]).rvs(size=n0)[:, None]
    age = np.repeat(0, n0)[:, None]
    
    # at time 0
    nt = np.copy(n0)
    zt = np.copy(z0)
    
    current_time = 0
    
    while((np.shape(zt)[0] < popu_max) and (current_time != total_time)):
        # initialize 
        current_n = np.shape(zt)[0]
        rep_stalks = np.repeat(np.nan, current_n)[:, None]
        surv = np.repeat(np.nan, current_n)[:, None]
        zprime = np.repeat(np.nan, current_n)[:, None]
        num_recruits = 0 
        if year==True:
            x_year = np.repeat(current_year, age.shape[0])
            X_t = np.concatenate((zt, age, x_year[:, None]), axis=1)
        else: 
            X_t = np.concatenate((zt, age), axis=1)
        
        # we simulate those breeders first
        # Simulating whether breeding by generating binomial random number 
        # will get a 1 if breeding
        p_breeding = models["m_fec"].predict_y(X_t)[0]
        rep_breeding = np.random.binomial(n = 1, p = p_breeding)
        whether_bre = (rep_breeding == 1).reshape(1, current_n)[0]          
        num_bre = np.sum(whether_bre)
        num_nonbre = current_n - num_bre
        
        # for those breeding individuals, we simulate the numbers of flowering 
        # stalks with Negative Binomial or Poisson distributions
        if NB == True:
            nb_predic = models["m_flow_nb"].predict_y(X_t[whether_bre, :])
            p_nb = nb_predic[0] / nb_predic[1]
            n_nb = nb_predic[0] * p_nb / (1 - p_nb)
            rep_stalks[whether_bre] = np.random.negative_binomial(n=n_nb, p=p_nb) + 1
        else: 
            lambda_t = models["m_flow_poi"].predict_y(X_t[whether_bre, :])[0]
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

        # store the simulation data
        # Here: age is for the age at current not ageNext
        data = pd.DataFrame(np.concatenate((zt, zprime, rep_breeding, rep_stalks, 
                                            surv, age, p_breeding, p_surv, 
                                            np.repeat(current_time, current_n)[:, None]), axis=1),
                            columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'p_fec', 'p_surv', 'time'])
        
        if(current_time == 0):
            IBM_data = pd.DataFrame.copy(data)
        else:
            IBM_data = pd.concat([IBM_data, data]).reset_index(drop=True)
        
        
        # print updated information
        print(current_time, "\n ")
        
        # update population
        if (num_bre != 0):
            zt = np.concatenate((zprime[whether_surv], rep_size))
            age = np.concatenate((age[whether_surv]+1, np.repeat(0, num_recruits)[:, None]))
        else:
            age = age+1
        
        current_time = current_time+1
        
        
    return(IBM_data)

def IBM_1step(zt, age, models, NB=False, year=False, current_year=np.nan):
    # If year=True, this means that the models contains year as another covariate
    if (year==True) and (np.isnan(current_year)):
        assert False, '\n Please indicate the current_year when considering considering year.'

    # initialize 
    current_n = np.shape(zt)[0]
    rep_stalks = np.repeat(np.nan, current_n)[:, None]
    surv = np.repeat(np.nan, current_n)[:, None]
    zprime = np.repeat(np.nan, current_n)[:, None]
    ageprime = np.repeat(np.nan, current_n)[:, None]
    num_recruits = 0 
    zt = np.array(zt)[:, None]
    age = np.array(age)[:, None]

    if year==True:
        x_year = np.repeat(current_year, age.shape[0])
        X_t = np.concatenate((zt, age, x_year[:, None]), axis=1)
    else: 
        X_t = np.concatenate((zt, age), axis=1)

    # we simulate those breeders first
    p_breeding = models["m_fec"].predict_y(X_t)[0]
    rep_breeding = np.random.binomial(n = 1, p = p_breeding)
    whether_bre = (rep_breeding == 1).reshape(1, current_n)[0]        
    num_bre = np.sum(whether_bre)
    num_nonbre = current_n - num_bre
        
    # for those breeding individuals, we simulate the numbers of flowering 
    # stalks with Negative Binomial or Poisson distributions
    if NB == True:
        nb_predic = models["m_flow_nb"].predict_y(X_t[whether_bre, :])
        
        p_nb = nb_predic[0] / nb_predic[1]
        n_nb = nb_predic[0] * p_nb / (1 - p_nb)
        rep_stalks[whether_bre] = np.random.negative_binomial(n=n_nb, p=p_nb) + 1
    else: 
        lambda_t = models["m_flow_poi"].predict_y(X_t[whether_bre, :])[0]
        if np.any(np.array(lambda_t) >= 10000):
            warnings.warn('lambda_t is too large !')
            rep_stalks[whether_bre] = 0
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
    
    ageprime[whether_surv] = age[whether_surv]+1

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
        p_breeding = np.concatenate((p_breeding, np.repeat(np.nan, num_recruits)[:, None]))
        p_surv = np.concatenate((p_surv, np.repeat(np.nan, num_recruits)[:, None]))   
    
    data = pd.DataFrame(np.concatenate((zt, zprime, rep_breeding, rep_stalks, 
                                        surv, age, ageprime, p_breeding, p_surv), axis=1),
                        columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext', 'p_fec', 'p_surv'])    

    return(data)

def IBM_1step_id(zt, age, id, models, NB=False, year=False, current_year=np.nan):
    # If year=True, this means that the models contains year as another covariate
    if (year==True) and (np.isnan(current_year)):
        assert False, '\n Please indicate the current_year when considering considering year.'
    # initialize 
    current_n = np.shape(zt)[0]
    rep_stalks = np.repeat(np.nan, current_n)[:, None]
    surv = np.repeat(np.nan, current_n)[:, None]
    zprime = np.repeat(np.nan, current_n)[:, None]
    ageprime = np.repeat(np.nan, current_n)[:, None]
    num_recruits = 0 
    zt = np.array(zt)[:, None]
    age = np.array(age)[:, None]
    id = np.array(id)[:, None]
    
    if year==True:
        x_year = np.repeat(current_year, age.shape[0])
        X_t = np.concatenate((zt, age, x_year[:, None]), axis=1)
    else: 
        X_t = np.concatenate((zt, age), axis=1)

    # we simulate those breeders first
    p_breeding = models["m_fec"].predict_y(X_t)[0]
    rep_breeding = np.random.binomial(n = 1, p = p_breeding)
    whether_bre = (rep_breeding == 1).reshape(1, current_n)[0]        
    num_bre = np.sum(whether_bre)
    num_nonbre = current_n - num_bre
        
    # for those breeding individuals, we simulate the numbers of flowering 
    # stalks with Negative Binomial or Poisson distributions
    if NB == True:
        nb_predic = models["m_flow_nb"].predict_y(X_t[whether_bre, :])
        
        p_nb = nb_predic[0] / nb_predic[1]
        n_nb = nb_predic[0] * p_nb / (1 - p_nb)
        rep_stalks[whether_bre] = np.random.negative_binomial(n=n_nb, p=p_nb) + 1
    else: 
        lambda_t = models["m_flow_poi"].predict_y(X_t[whether_bre, :])[0]

        if np.any(np.isnan(lambda_t)):
            warnings.warn('nan lambda_t produced!!')
            data = pd.DataFrame(0, index=[0], columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext']) 

            return(data)

        if np.any(np.array(lambda_t) >= 10000):
            warnings.warn('lambda_t is too large !!')
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
    
    ageprime[whether_surv] = age[whether_surv]+1

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
        idprime = np.concatenate((id, np.array([str(uuid.uuid4()) for _ in range(num_recruits)])[:, None]))
        p_breeding = np.concatenate((p_breeding, np.repeat(np.nan, num_recruits)[:, None]))
        p_surv = np.concatenate((p_surv, np.repeat(np.nan, num_recruits)[:, None]))   
    
    data = pd.DataFrame(np.concatenate((zt, zprime, rep_breeding, rep_stalks, 
                                        surv, age, ageprime, p_breeding, p_surv), axis=1),
                        columns=['size', 'sizeNext', 'fec', 'flow', 'surv', 'age', 'ageNext', 'p_fec', 'p_surv']) 
    data['ID'] = idprime  

    return(data)

# define a funtion to return prediction inputs
def full_dataset_generator(lower=-0.5, upper=4.5, m_size=100, max_age=14):
    h = (upper - lower) / m_size
    mesh = (np.arange(m_size) + 0.5)* h + lower
    age = np.array(range(0, max_age + 1))
    Xp = np.array([(x, y) for x in mesh for y in age])
    r_data = pd.DataFrame({'size': Xp[:, 0], 'age': Xp[:, 1]})
    return r_data


# Setting up kernel matrix including mesh points and mesh width.
def kernel_setting(m_size, lower, upper, max_age):
    h = (upper - lower) / m_size
    mesh = (np.arange(m_size) + 0.5)* h + lower
    kernel_number = max_age + 1
    # sometimes k_number should be max_age + 2  
    # adding 1 for age zero, adding 1 for maximum. But here, we are constructing
    # a “pre-reproductive Kernel”, so the minimum age should be
    # 1 rather than 0. Therefore, k_number <- max_age + 1  
    # adding 1 for those older than the maximum.
    return([mesh, h, kernel_number, m_size])
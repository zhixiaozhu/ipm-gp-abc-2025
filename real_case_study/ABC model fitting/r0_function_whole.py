from r0_function_seperate import *

########################################################################################################################################################################################################
# when years are fitted altogether

# define a naming vectors containing all the parameters' names 
para_names_whole = ['sur_lsize', 'sur_lage', 'sur_ltmax', 'sur_ltmin', 'sur_lprecip', 'sur_kvar',
                    'grwf_lsize', 'grwf_lage','grwf_ltmax','grwf_ltmin', 'grwf_lprecip', 'grwf_kvar', 'grwf_lvar',
                    'grwnf_lsize', 'grwnf_lage', 'grwnf_ltmax', 'grwnf_ltmin', 'grwnf_lprecip', 'grwnf_kvar', 'grwnf_lvar',
                    'fec_lsize', 'fec_lage', 'fec_ltmax', 'fec_ltmin','fec_lprecip','fec_kvar',
                    'flowp_lsize', 'flowp_lage','flowp_ltmax','flowp_ltmin', 'flowp_lprecip', 'flowp_kvar',
                    'alpha', 'beta', 'recruit_p']


def extract_parameters_whole(models):
    theta = pd.DataFrame(0, range(1), para_names_whole)
    theta['alpha'] = models['alpha']
    theta['beta'] = models['beta']
    theta['recruit_p'] = models['recruit_p']

    theta['sur_lsize'], theta['sur_lage'], theta['sur_ltmax'], theta['sur_ltmin'], theta['sur_lprecip'] = np.array(models['m_sur'].kernel.lengthscales)
    theta['sur_kvar'] = np.array(models['m_sur'].kernel.variance)

    theta['grwf_lsize'], theta['grwf_lage'], theta['grwf_ltmax'], theta['grwf_ltmin'], theta['grwf_lprecip'] = np.array(models['m_grw_f'].kernel.lengthscales)
    theta['grwf_kvar'] = np.array(models['m_grw_f'].kernel.variance); theta['grwf_lvar'] = np.array(models['m_grw_f'].likelihood.variance) 
    
    theta['grwnf_lsize'], theta['grwnf_lage'], theta['grwnf_ltmax'], theta['grwnf_ltmin'], theta['grwnf_lprecip'] = np.array(models['m_grw_nf'].kernel.lengthscales)
    theta['grwnf_kvar'] = np.array(models['m_grw_nf'].kernel.variance); theta['grwnf_lvar'] = np.array(models['m_grw_nf'].likelihood.variance) 

    theta['fec_lsize'], theta['fec_lage'], theta['fec_ltmax'], theta['fec_ltmin'], theta['fec_lprecip'] = np.array(models['m_fec'].kernel.lengthscales)
    theta['fec_kvar'] = np.array(models['m_fec'].kernel.variance)

    theta['flowp_lsize'], theta['flowp_lage'], theta['flowp_ltmax'], theta['flowp_ltmin'], theta['flowp_lprecip'] = np.array(models['m_flow_poi'].kernel.lengthscales)
    theta['flowp_kvar'] = np.array(models['m_flow_poi'].kernel.variance)

    return theta


def baseline_model_fit_whole(data_whole, NB=False):
    data_ori = data_whole.copy()

    ############################################################################################################################################################# 
    # datasets constructing

    # 1 - surv
    index_sur = np.logical_not(np.isnan(data_ori["surv"]))
    size_sur = data_ori["size"][index_sur].to_numpy()
    surv_sur = data_ori["surv"][index_sur].to_numpy()
    age_sur = data_ori["age"][index_sur].to_numpy()
    tmax_sur = data_ori["tmax"][index_sur].to_numpy()
    tmin_sur = data_ori["tmin"][index_sur].to_numpy()
    precip_sur = data_ori["precip"][index_sur].to_numpy()  

    # 2 - growth
    # i - if we conserding "fec" effect
    # data for breeders (fec == 1)
    index_grw = np.logical_not(np.isnan(data_ori["sizeNext"])) & np.logical_not(np.isnan(data_ori["size"]))
    fec_grw = data_ori["fec"][index_grw]

    size_grw_f = data_ori["size"][index_grw][fec_grw == 1].to_numpy()
    sizeNext_grw_f = data_ori["sizeNext"][index_grw][fec_grw == 1].to_numpy()
    age_grw_f = data_ori["age"][index_grw][fec_grw == 1].to_numpy()
    tmax_grw_f = data_ori["tmax"][index_grw][fec_grw == 1].to_numpy()
    tmin_grw_f = data_ori["tmin"][index_grw][fec_grw == 1].to_numpy()
    precip_grw_f = data_ori["precip"][index_grw][fec_grw == 1].to_numpy() 


    # data for breeders (fec == 0)
    size_grw_nf = data_ori["size"][index_grw][fec_grw == 0].to_numpy()
    sizeNext_grw_nf = data_ori["sizeNext"][index_grw][fec_grw == 0].to_numpy()
    age_grw_nf = data_ori["age"][index_grw][fec_grw == 0].to_numpy()
    tmax_grw_nf = data_ori["tmax"][index_grw][fec_grw == 0].to_numpy()
    tmin_grw_nf = data_ori["tmin"][index_grw][fec_grw == 0].to_numpy()
    precip_grw_nf = data_ori["precip"][index_grw][fec_grw == 0].to_numpy() 

    # 3 - fec
    index_fec = np.logical_not(np.isnan(data_ori["fec"]))
    size_fec = data_ori["size"][index_fec].to_numpy()
    fec_fec = data_ori["fec"][index_fec].to_numpy()
    age_fec = data_ori["age"][index_fec].to_numpy()
    tmax_fec = data_ori["tmax"][index_fec].to_numpy()
    tmin_fec = data_ori["tmin"][index_fec].to_numpy()
    precip_fec = data_ori["precip"][index_fec].to_numpy() 
    random_index = np.random.choice(500, 50)

    # 4 - number of flowering stalks
    index_flow = data_ori["fec"] == 1
    size_flow = data_ori["size"][index_flow].to_numpy()
    flow_flow = data_ori["flow"][index_flow].to_numpy()
    age_flow = data_ori["age"][index_flow].to_numpy()
    tmax_flow = data_ori["tmax"][index_flow].to_numpy()
    tmin_flow = data_ori["tmin"][index_flow].to_numpy()
    precip_flow = data_ori["precip"][index_flow].to_numpy() 
    # 5 - recruit size (indp with the parent size)
    index_recr = np.isnan(data_ori["size"])
    sizeNext_recr = data_ori["sizeNext"][index_recr].to_numpy()

    # log have already taken for these simulated datasets
    # 1 - surv
    X_sur = np.concatenate((size_sur[:, None], age_sur[:, None], tmax_sur[:, None], tmin_sur[:, None], precip_sur[:, None]), axis=1)
    Y_sur = surv_sur[:, None]

    # 2 - growth
    X_grw_f = np.concatenate((size_grw_f[:, None], age_grw_f[:, None], tmax_grw_f[:, None], tmin_grw_f[:, None], precip_grw_f[:, None]), axis=1)
    Y_grw_f = sizeNext_grw_f[:, None]
    X_grw_nf = np.concatenate((size_grw_nf[:, None], age_grw_nf[:, None], tmax_grw_nf[:, None], tmin_grw_nf[:, None], precip_grw_nf[:, None]), axis=1)
    Y_grw_nf = sizeNext_grw_nf[:, None]

    # 3 - fec
    X_fec = np.concatenate((size_fec[:, None], age_fec[:, None], tmax_fec[:, None], tmin_fec[:, None], precip_fec[:, None]), axis=1)
    Y_fec = fec_fec[:, None]

    # 4 - number of flowering stalks
    X_flow = np.concatenate((size_flow[:, None], age_flow[:, None], tmax_flow[:, None], tmin_flow[:, None], precip_flow[:, None]), axis=1)
    Y_flow = flow_flow[:, None] - 1
    
    ############################################################################################################################################################# 
    # model fitting

    # 1 - survive 
    l_sur = gpflow.likelihoods.Bernoulli()
    k_sur = gpflow.kernels.RBF(lengthscales=np.array([1,1,1,1,1]))

    m_sur1 = gpflow.models.VGP((X_sur, Y_sur), kernel=k_sur, likelihood=l_sur)

    opt_sur = gpflow.optimizers.Scipy()
    opt_sur.minimize(m_sur1.training_loss, variables=m_sur1.trainable_variables, options=dict(maxiter=10000))

    # 2 - growth: if we conserding "fec" effect
    # growth model fitting for breeders (fec == 1)

    k_grw_f = gpflow.kernels.RBF(lengthscales=np.array([1,1,1,1,1]))

    m_grw_f = gpflow.models.GPR(data=(X_grw_f, Y_grw_f), 
                                kernel=k_grw_f, mean_function=None)

    opt_grw_f = gpflow.optimizers.Scipy()
    opt_grw_f.minimize(m_grw_f.training_loss, variables=m_grw_f.trainable_variables, options=dict(maxiter=10000))


    # growth model fitting for the remainings (fec == 0)
    k_grw_nf = gpflow.kernels.RBF(lengthscales=np.array([1,1,1,1,1]))

    m_grw_nf = gpflow.models.GPR(data=(X_grw_nf, Y_grw_nf), 
                                kernel=k_grw_nf, mean_function=None)

    opt_grw_nf = gpflow.optimizers.Scipy()
    opt_grw_nf.minimize(m_grw_nf.training_loss, variables=m_grw_nf.trainable_variables, options=dict(maxiter=10000))


    # 3 - fec
    l_fec = gpflow.likelihoods.Bernoulli()
    k_fec = gpflow.kernels.RBF(lengthscales=np.array([1,990,1,1,1]), variance=6)
    m_fec = gpflow.models.VGP((X_fec, Y_fec), kernel=k_fec, likelihood=l_fec)

    opt_fec = gpflow.optimizers.Scipy()
    opt_fec.minimize(m_fec.training_loss, variables=m_fec.trainable_variables, options=dict(maxiter=10000))



    # 4 - number of flowering stalks 
    # (i) Poisson
    l_flow_poi = gpflow.likelihoods.Poisson()
    k_flow_poi = gpflow.kernels.RBF(lengthscales=np.array([1,1,1,1,1]))

    m_flow_poi = gpflow.models.VGP((X_flow, Y_flow), 
                                kernel=k_flow_poi, likelihood=l_flow_poi)
    opt_flow_poi = gpflow.optimizers.Scipy()
    result = opt_flow_poi.minimize(m_flow_poi.training_loss, variables=m_flow_poi.trainable_variables, 
                        method="L-BFGS-B", options=dict(maxiter=10000))


    # 4 - number of flowering stalks 
    #  (ii) NB

    if NB == True:
        assert False, '\n NB = True is not supported !'
    else:
        m_flow_nb = None


    # 6 - recruits' establishment probability
    # there is no information about the number of seed in the data set.
    # we may use the #offspring/#flowering stalks to estimate it.
    mean_log_recr = np.mean(sizeNext_recr)
    var_log_recr = np.var(sizeNext_recr)
    v = var_log_recr * sizeNext_recr.shape[0] / (sizeNext_recr.shape[0] - 1)
    # for gamma distribution
    alpha = np.power(mean_log_recr, 2) / v
    beta = mean_log_recr / v
    recruit_estab = sum(np.isnan(data_ori["size"])) / data_ori["flow"].sum()
    m_recruits = pd.DataFrame([alpha,beta,recruit_estab], 
                            index=['gamma_alpha','gamma_beta','recruit_estab'],
                            columns=['values'])

    models = {
        "m_sur": m_sur1,
        "m_grw_f": m_grw_f,
        "m_grw_nf": m_grw_nf,
        "m_fec": m_fec,
        "m_flow_poi": m_flow_poi,
        "m_flow_nb": m_flow_nb,
        "alpha": m_recruits.iloc[0,0],
        "beta": m_recruits.iloc[1,0],
        "recruit_p": m_recruits.iloc[2,0]
    }

    return models



# our main function and class.
#  IPM_whole is standing for the big IPM build for all years 
class ipmmcmc_whole():
    def __init__(self, popu_dataset_whole, popu_0toT, z0, tmax_list, tmin_list, precip_list, alpha, beta, recruit_p):
        # z0: the initial population structure
        self.popu_dataset_whole = popu_dataset_whole
        self.popu_0toT = popu_0toT
        self.z0 = z0
        self.tmax_list = tmax_list
        self.tmin_list = tmin_list
        self.precip_list = precip_list
        self.alpha = alpha
        self.beta = beta
        self.recruit_p = recruit_p

    # build IPM_whole by given random indeces for each vital rate's MCMC sample (in parallel computing)
    def random_IPM_whole_para(self, total_samples, given_theta=False, given_weight=False, random=True):

        result = []
        n_actor = multiprocessing.cpu_count()-1
        simulators = [para_ipmmcmc_whole.remote(self) for _ in range(n_actor)]
        if given_theta==False and given_weight==False and random==True:
            for i in np.arange(0, total_samples, n_actor):
                result.extend(ray.get([s.random_IPM_whole.remote() for s in simulators]))
        elif given_theta==True and given_weight==False and random==True:
            for i in np.arange(0, total_samples, n_actor):
                result.extend(ray.get([s.random_IPM_ABC.remote() for s in simulators])) 
        elif given_theta==True and given_weight==True and random==True:
            # gievn_theta==True, given weight==True
            for i in np.arange(0, total_samples, n_actor):
                result.extend(ray.get([s.random_IPM_ABC_weight.remote() for s in simulators]))
        elif given_theta==False and given_weight==False and random==False:
            for i in np.arange(0, total_samples, n_actor):
                result.extend(ray.get([s.not_random_IPM_ABC.remote(i+k) for k,s in enumerate(simulators)]))     
        elif given_theta==True and given_weight==True and random==False: 
            for i in np.arange(0, total_samples, n_actor):
                result.extend(ray.get([s.not_random_IPM_ABC_weight.remote() for s in simulators]))            

        return result
    
    # build IPM_whole by given random index for each vital rate's MCMC sample (in series computing)
    def random_IPM_whole(self):
        index = np.random.randint(low=0, high=[ci_niter(self.samples_sur[0].shape[0]),
                                               ci_niter(self.samples_grw_f[0].shape[0]),
                                               ci_niter(self.samples_grw_nf[0].shape[0]),
                                               ci_niter(self.samples_fec[0].shape[0]),
                                               ci_niter(self.samples_flow[0].shape[0])], size=5)

        return self.IPM_whole_givenindex(index)
    
    # still random, but not completely random as the previous. It is not sampling particle from the previous ABC-SMC step
    def random_IPM_ABC(self):
        # randomly select a particle from the previous step.
        index = self.p_index[np.random.randint(0, self.p_index.shape[0])]
        return self.IPM_whole_givenindex(index)

    # not random. build IPM_whole based on a given index
    def not_random_IPM_ABC(self, i):
        if i < self.p_index.shape[0]:
            index = self.p_index[i]
            return self.IPM_whole_givenindex(index)  


    # compared with random_IPM_ABC, not_random_IPM_ABC_weight is now drawing index accordingly to a 'weight' vector.
    def not_random_IPM_ABC_weight(self):
        index = self.p_index[np.random.choice(self.p_index.shape[0], p=self.weight)].copy()
        return self.IPM_whole_givenindex(index) 
    

    # compared with not_random_IPM_ABC_weight,  one of the five vital rates is going to be replaced by a random index.
    def random_IPM_ABC_weight(self):
        # randomly select a particle from the previous step.
        index = self.p_index[np.random.choice(self.p_index.shape[0], p=self.weight)].copy()
        index[np.random.randint(0, 5)] = np.random.randint(0, 5000) 

        return self.IPM_whole_givenindex(index) 


    def IPM_whole_givenindex(self, index):
        # sur
        for var, var_samples in zip(self.hmc_helper_sur.current_state, self.samples_sur):
            var.assign(var_samples[index[0]])
        # grw_f
        for var, var_samples in zip(self.hmc_helper_grw_f.current_state, self.samples_grw_f):
            var.assign(var_samples[index[1]])
        # grw_nf
        for var, var_samples in zip(self.hmc_helper_grw_nf.current_state, self.samples_grw_nf):
            var.assign(var_samples[index[2]])
        # fec
        for var, var_samples in zip(self.hmc_helper_fec.current_state, self.samples_fec):
            var.assign(var_samples[index[3]])
        # flow
        for var, var_samples in zip(self.hmc_helper_flow.current_state, self.samples_flow):
            var.assign(var_samples[index[4]])

        # cache (used for cholesky decomposition) for sampled MCMC are calcualted and stored
        #  the index for the MCMC samples are consistent with the corresponding stored files' name
        self.m_sur_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/sur/{index[0]}", mode="rb"))
        self.m_grw_f_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/grw_f/{index[1]}", mode="rb")) 
        self.m_grw_nf_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/grw_nf/{index[2]}", mode="rb")) 
        self.m_fec_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/fec/{index[3]}", mode="rb")) 
        self.m_flow_poi_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/flow/{index[4]}", mode="rb")) 

        models_now2 = {
            "m_sur": self.m_sur_new ,
            "m_grw_f": self.m_grw_f_new,
            "m_grw_nf": self.m_grw_nf_new ,
            "m_fec": self.m_fec_new,
            "m_flow_poi": self.m_flow_poi_new,
            "alpha": self.alpha,
            "beta": self.beta,
            "recruit_p": self.recruit_p
        }

        s = []
        for _ in range(self.rep):
            c_s = []
            # updating yearly parameters for recuruitments.
            models_now2['alpha'] = self.recru_full['alpha'][0]
            models_now2['beta'] = self.recru_full['beta'][0]
            models_now2['recruit_p'] = self.recru_full['recruit_p'][0]

            data_simu = IBM_1step_cashe(zt=self.z0[0], age=self.z0[1], models=models_now2, env=True, 
                                        tmax=self.tmax_list[0], tmin=self.tmin_list[0], precip=self.precip_list[0])
            if (data_simu.shape[0] == 0) or (data_simu.shape[0] == 1):
                return (index, [[np.repeat(np.inf, 5) for _ in range(len(self.popu_0toT))]])
            current_s = list_comparisons_interested(exp_1step_data=self.popu_0toT[0], simu_1step_data=data_simu)
            c_s.append(current_s)

            for t in range(1, len(self.popu_0toT)):

                # updating yearly parameters for recuruitments.
                models_now2['alpha'] = self.recru_full['alpha'][t]
                models_now2['beta'] = self.recru_full['beta'][t]
                models_now2['recruit_p'] = self.recru_full['recruit_p'][t]

                z = popu_structure_noid(data_simu) 
                data_simu = IBM_1step_cashe(zt=z[0], age=z[1], models=models_now2, env=True, 
                                            tmax=self.tmax_list[t], tmin=self.tmin_list[t], precip=self.precip_list[t])

                if (data_simu.shape[0] == 0) or (data_simu.shape[0] == 1):
                    return (index, [[np.repeat(np.inf, 5) for _ in range(len(self.popu_0toT))]])
                
                try:
                    current_s = list_comparisons_interested(exp_1step_data=self.popu_0toT[t], simu_1step_data=data_simu)
                except Exception as e:
                    now = datetime.now().strftime("%H_%M_%S")
                    pickle.dump(z , open(file = os.getcwd() + f"/error/z{t}" + now + ".pkl", mode="wb"))
                    pickle.dump(data_simu , open(file = os.getcwd() + "/error/data_simu" + now + ".pkl", mode="wb"))
                    pickle.dump(index, open(file = os.getcwd() + "/error/index" + now + ".pkl", mode="wb"))
                    print(f'Error found at t={t}', flush=True)
                    sys.exit(e)                     
                c_s.append(current_s)

            s.append(c_s)

        return (index, s)
    
    # a function used for cache information ONLY
    def GPmodels_givenindex_cache(self, index):
        # sur
        for var, var_samples in zip(self.hmc_helper_sur.current_state, self.samples_sur):
            var.assign(var_samples[index[0]])
        # grw_f
        for var, var_samples in zip(self.hmc_helper_grw_f.current_state, self.samples_grw_f):
            var.assign(var_samples[index[1]])
        # grw_nf
        for var, var_samples in zip(self.hmc_helper_grw_nf.current_state, self.samples_grw_nf):
            var.assign(var_samples[index[2]])
        # fec
        for var, var_samples in zip(self.hmc_helper_fec.current_state, self.samples_fec):
            var.assign(var_samples[index[3]])
        # flow
        for var, var_samples in zip(self.hmc_helper_flow.current_state, self.samples_flow):
            var.assign(var_samples[index[4]])

        models_now2 = {
            "m_sur": self.m_sur_new ,
            "m_grw_f": self.m_grw_f_new,
            "m_grw_nf": self.m_grw_nf_new ,
            "m_fec": self.m_fec_new,
            "m_flow_poi": self.m_flow_poi_new,
            "alpha": self.alpha,
            "beta": self.beta,
            "recruit_p": self.recruit_p
        }

    

    # build IPM_whole based on MLEs, calculating the corresponding summary statistics for the (MLE simulation) VS (observation or another MLE simulation)
    def IPM_whole_mle(self, target):

        if target != 'MLE' and target != 'observations':
            assert False, '\n We are still working hard to develop this function! Please come back later.'
         
        models_now2 = {
            "m_sur": self.models_mle['m_sur'],
            "m_grw_f": self.models_mle['m_grw_f'],
            "m_grw_nf": self.models_mle['m_grw_nf'],
            "m_fec": self.models_mle['m_fec'],
            "m_flow_poi": self.models_mle['m_flow_poi'],
            "alpha": self.alpha,
            "beta": self.beta,
            "recruit_p": self.recruit_p
        }

        s = []
        for _ in range(self.rep):
            c_s = []
            # updating yearly parameters for recuruitments.
            models_now2['alpha'] = self.recru_full['alpha'][0]
            models_now2['beta'] = self.recru_full['beta'][0]
            models_now2['recruit_p'] = self.recru_full['recruit_p'][0]

            
            data_simu1 = IBM_1step(zt=self.z0[0], age=self.z0[1], models=models_now2, env=True, 
                                  tmax=self.tmax_list[0], tmin=self.tmin_list[0], precip=self.precip_list[0])
            
            if target == 'MLE':
                # generate two simulated datasets each time
                data_simu2 = IBM_1step(zt=self.z0[0], age=self.z0[1], models=models_now2, env=True, 
                                    tmax=self.tmax_list[0], tmin=self.tmin_list[0], precip=self.precip_list[0])
                
                current_s = list_comparisons_interested(exp_1step_data=data_simu1, simu_1step_data=data_simu2)
            else:
                current_s = list_comparisons_interested(exp_1step_data=self.popu_0toT[0], simu_1step_data=data_simu1) 
            
            c_s.append(current_s) 

            for t in range(1, len(self.popu_0toT)):

                # updating yearly parameters for recuruitments.
                models_now2['alpha'] = self.recru_full['alpha'][t]
                models_now2['beta'] = self.recru_full['beta'][t]
                models_now2['recruit_p'] = self.recru_full['recruit_p'][t]

                z1 = popu_structure_noid(data_simu1) 
                

                data_simu1 = IBM_1step(zt=z1[0], age=z1[1], models=models_now2, env=True, 
                                      tmax=self.tmax_list[t], tmin=self.tmin_list[t], precip=self.precip_list[t])
                
                if target == 'MLE':
                    z2 = popu_structure_noid(data_simu2)  
                    data_simu2 = IBM_1step(zt=z2[0], age=z2[1], models=models_now2, env=True, 
                                        tmax=self.tmax_list[t], tmin=self.tmin_list[t], precip=self.precip_list[t])

                    current_s = list_comparisons_interested(exp_1step_data=data_simu1, simu_1step_data=data_simu2)
                else:
                    current_s = list_comparisons_interested(exp_1step_data=self.popu_0toT[t], simu_1step_data=data_simu1) 
                
                c_s.append(current_s)

            s.append(c_s)

        return s


    # predictions generated by MLEs
    def IPM_prediction_mle(self):

        models_now2 = {
            "m_sur": self.models_mle['m_sur'],
            "m_grw_f": self.models_mle['m_grw_f'],
            "m_grw_nf": self.models_mle['m_grw_nf'],
            "m_fec": self.models_mle['m_fec'],
            "m_flow_poi": self.models_mle['m_flow_poi'],
            "alpha": self.alpha,
            "beta": self.beta,
            "recruit_p": self.recruit_p
        }

        n_all = []; n1_all = []; n2_all = []
        for _ in range(self.rep):
            n_size = []; n1_size = []; n2_size = []
            # updating yearly parameters for recuruitments.
            models_now2['alpha'] = self.recru_full['alpha'][0]
            models_now2['beta'] = self.recru_full['beta'][0]
            models_now2['recruit_p'] = self.recru_full['recruit_p'][0]

            
            data_simu1 = IBM_1step(zt=self.z0[0], age=self.z0[1], models=models_now2, env=True, 
                                  tmax=self.tmax_list[0], tmin=self.tmin_list[0], precip=self.precip_list[0])
            
            n_new1 = np.sum(data_simu1["surv"] == 1)
            n_new2 = np.sum(np.isnan(data_simu1["size"])) 
            n_size.append(n_new1+n_new2); n1_size.append(n_new1); n2_size.append(n_new2) 

            for t in range(1, 9):

                # updating yearly parameters for recuruitments.
                models_now2['alpha'] = self.recru_full['alpha'][t]
                models_now2['beta'] = self.recru_full['beta'][t]
                models_now2['recruit_p'] = self.recru_full['recruit_p'][t]

                z1 = popu_structure_noid(data_simu1) 
                

                data_simu1 = IBM_1step(zt=z1[0], age=z1[1], models=models_now2, env=True, 
                                      tmax=self.tmax_list[t], tmin=self.tmin_list[t], precip=self.precip_list[t])

                n_new1 = np.sum(data_simu1["surv"] == 1)
                n_new2 = np.sum(np.isnan(data_simu1["size"])) 
                n_size.append(n_new1+n_new2); n1_size.append(n_new1); n2_size.append(n_new2)

            n_all.append(n_size)
            n1_all.append(n1_size)
            n2_all.append(n2_size)

        return n_all, n1_all, n2_all



    # algo_continue: Boolean, optional: Whether run is a continuation of an earlier run.
    # Notice that, algo_continue=True has NOT been tested yet. 
    # Pass this with the model_name argument to automatically load previous history 
    # and crossover probability files. Default: False
    def ABC_SMC(self, quantiles, n_particles, details=True,
                algo_continue=False, hist_mad=None, hist_threshold=None, hist_index=None, hist_weight=None):
        # only generate a single population for a single theta sample
        if isinstance(n_particles, int):
            sys.exit("n_particle should be an array or a list! ")

        if (len(quantiles)) != len(n_particles):
            sys.exit("len(quantiles) should equal to len(n_particles)")


        if (algo_continue==True) and ((hist_mad==None) or (hist_threshold==None) or (hist_index==None) or (hist_weight==None)):
            sys.exit("Please provide all the history files to continue th algorithm.")

        self.rep = 1

        if algo_continue == False:
            # for SMC iteration c=0
            print('c=0 ' + time.strftime("%H:%M:%S", time.localtime()), flush=True)
            particles = np.array(self.random_IPM_whole_para(total_samples=n_particles[0], given_theta=False), dtype=object)
            particles_summary = np.array([item for sublist in particles[:, 1] for item in sublist])
            particles_index = np.array([sublist for sublist in particles[:, 0]]); del(particles) # reliease memory

            if details == True:
                np.save(open(file = os.getcwd()+"/ABC_details/p_index_ini.pkl", mode="wb"), particles_index)

            # reshape
            c_shape = particles_summary.shape
            particles_summary = particles_summary.reshape((c_shape[0], c_shape[1]*c_shape[2]))
            self.mad = np.nanmedian(np.absolute(particles_summary - np.nanmedian(particles_summary, axis=0)), axis=0)
            s_mean = np.nanmean(particles_summary, axis=0)
            s_median = np.nanmedian(particles_summary, axis=0)
            dis = np.sqrt(np.sum((particles_summary/self.mad)**2, axis=1))

            # we accept all the particles at c=0 
            self.threshold = np.nanquantile(dis, axis=0, q=quantiles[0])
            accepted = np.repeat(True, particles_index.shape[0]) 

            self.p_index = particles_index[accepted]
            # equal weights
            self.weight = np.repeat(1/np.sum(accepted), np.sum(accepted))

            print(f'Total: {particles_index.shape[0]}', flush=True)
            print(f'Left: {self.p_index.shape[0]}', flush=True)
            print(f'Unique: {np.unique(self.p_index , axis=0).shape[0]} \n', flush=True)

            if details == True:
                np.save(open(file = os.getcwd()+f"/ABC_details/particles_summary{0}.pkl", mode="wb"), particles_summary)
                np.save(open(file = os.getcwd()+f"/ABC_details/accepted{0}.pkl", mode="wb"), accepted)
                np.save(open(file = os.getcwd()+f"/ABC_details/p_index_{0}.pkl", mode="wb"), self.p_index)
                np.save(open(file = os.getcwd()+f"/ABC_details/threshold{0}.pkl", mode="wb"), self.threshold)
                np.save(open(file = os.getcwd()+f"/ABC_details/weight{0}.pkl", mode="wb"), self.weight)
                np.save(open(file = os.getcwd()+f"/ABC_details/mad{0}.pkl", mode="wb"), self.mad)
                np.save(open(file = os.getcwd()+f"/ABC_details/r_s_mean{0}.pkl", mode="wb"), s_mean)
                np.save(open(file = os.getcwd()+f"/ABC_details/r_s_median{0}.pkl", mode="wb"), s_median)
            del(particles_summary); del(particles_index); del(c_shape)

        else:
            self.p_index = np.load(open(file = hist_index, mode="rb"))
            self.weight = np.load(open(file = hist_weight, mode="rb"))
            self.mad = np.load(open(file = hist_mad, mode="rb"))
            self.threshold = np.load(open(file = hist_threshold, mode="rb"))

        # for SMC iteration c>0
        for c in range(1, len(quantiles)):
            print(f'c={c} ' + time.strftime("%H:%M:%S", time.localtime()), flush=True)

            n_alive_particles=0
            c_summary = []
            c_index = []
            total_summary = []

            while n_alive_particles<n_particles[c]:
                particles = np.array(self.random_IPM_whole_para(total_samples=10000,#np.minimum(n_particles[c]-n_alive_particles, 10000),
                                                                given_theta=True, given_weight=True), dtype=object)
                particles_summary = np.array([item for sublist in particles[:, 1] for item in sublist])
                particles_index = np.array([sublist for sublist in particles[:, 0]]); del(particles)
                # reshape
                c_shape = particles_summary.shape
                particles_summary = particles_summary.reshape((c_shape[0], c_shape[1]*c_shape[2]))

                # updating baselines
                dis = np.sqrt(np.sum((particles_summary/self.mad)**2, axis=1))

                # accept or reject based on the thershold computed from the previous step.
                accepted = dis <= self.threshold
                # update
                c_summary.extend(particles_summary[accepted])
                total_summary.extend(particles_summary)
                c_index.extend(particles_index[accepted]); del(particles_summary); del(particles_index); del(c_shape); del(dis)
                n_alive_particles = n_alive_particles+accepted.sum(); del(accepted)
                print('  '+time.strftime("%H:%M:%S", time.localtime()) + f' Required: {n_particles[c]}, Now: {n_alive_particles}', flush=True)

            t_summary = np.array([ii for ii in total_summary]); del(total_summary)
            s_mean = np.nanmean(t_summary, axis=0)
            s_median = np.nanmedian(t_summary, axis=0)
            self.mad = np.nanmedian(np.absolute(t_summary - np.nanmedian(t_summary, axis=0)), axis=0); del(t_summary)

            c_index2 = np.array([ii for ii in c_index]); del(c_index) # just tranform it from a list to an np.array
            c_weight = []
            for kk in range(c_index2.shape[0]):
                neighbour = np.sum(self.p_index == c_index2[kk], axis=1) >= 4
                c_weight.append(1/np.sum(self.weight[neighbour]))

            self.weight = c_weight/np.sum(c_weight); del(c_weight)
            self.p_index = c_index2; del(c_index2)
            summary = np.array([ii for ii in c_summary]); del(c_summary)
            dis_now = np.sqrt(np.sum((summary/self.mad)**2, axis=1))
            self.threshold = np.nanquantile(dis_now, axis=0, q=quantiles[c])

            print(f'Left: {self.p_index.shape[0]}', flush=True)
            print(f'Unique: {np.unique(self.p_index , axis=0).shape[0]} \n', flush=True)

            if details == True:
                np.save(open(file = os.getcwd()+f"/ABC_details/particles_summary{c}.pkl", mode="wb"), summary)
                np.save(open(file = os.getcwd()+f"/ABC_details/p_index_{c}.pkl", mode="wb"), self.p_index)
                np.save(open(file = os.getcwd()+f"/ABC_details/threshold{c}.pkl", mode="wb"), self.threshold)
                np.save(open(file = os.getcwd()+f"/ABC_details/weight{c}.pkl", mode="wb"), self.weight)
                np.save(open(file = os.getcwd()+f"/ABC_details/mad{c}.pkl", mode="wb"), self.mad)
                np.save(open(file = os.getcwd()+f"/ABC_details/r_s_mean{c}.pkl", mode="wb"), s_mean)
                np.save(open(file = os.getcwd()+f"/ABC_details/r_s_median{c}.pkl", mode="wb"), s_median)
            del(summary)

        return self.p_index, self.threshold



    def ABC_random(self, n_particles, details=True):

        self.rep = 1        

        # for SMC iteration c=0
        print('c=0 ' + time.strftime("%H:%M:%S", time.localtime()), flush=True)
        particles = np.array(self.random_IPM_whole_para(total_samples=n_particles[0], given_theta=False), dtype=object)
        particles_summary = np.array([item for sublist in particles[:, 1] for item in sublist])
        particles_index = np.array([sublist for sublist in particles[:, 0]]); del(particles) # reliease memory

        if details == True:
            np.save(open(file = os.getcwd()+"/ABC_details/p_index_ini.pkl", mode="wb"), particles_index)

        # reshape
        c_shape = particles_summary.shape
        particles_summary = particles_summary.reshape((c_shape[0], c_shape[1]*c_shape[2]))

        self.p_index = particles_index
        print(f'Total: {particles_index.shape[0]}', flush=True)
        print(f'Left: {self.p_index.shape[0]}', flush=True)
        print(f'Unique: {np.unique(self.p_index , axis=0).shape[0]} \n', flush=True)

        t = time.localtime()

        if details == True:
            np.save(open(file = os.getcwd()+f"/ABC_details/particles_summary{0}"+time.strftime("%H%M%S%d", t)+".pkl", mode="wb"), particles_summary)
            np.save(open(file = os.getcwd()+f"/ABC_details/p_index_{0}"+time.strftime("%H%M%S%d", t)+".pkl", mode="wb"), self.p_index)





    def simple_rejection(self, mad, threshold, total_n, rep, quantile_for_final, details=True):
        # only generate a single population for a single theta sample
        if not isinstance(total_n, int):
            sys.exit("total_n should be an integer! ")

        self.rep = 1        

        n_alive_particles = 0
        c_summary = []
        c_index = []
        total_summary = []

        while n_alive_particles < total_n:
            particles = np.array(self.random_IPM_whole_para(total_samples=10000,
                                                            given_theta=False, given_weight=False), dtype=object)
            particles_summary = np.array([item for sublist in particles[:, 1] for item in sublist])
            particles_index = np.array([sublist for sublist in particles[:, 0]]); del(particles)
            # reshape
            c_shape = particles_summary.shape
            particles_summary = particles_summary.reshape((c_shape[0], c_shape[1]*c_shape[2]))
            

            dis = np.sqrt(np.sum((particles_summary/mad)**2, axis=1))
            
            # accept or reject based on the thershold computed from the previous step.
            accepted = dis <= threshold
            # update
            c_summary.extend(particles_summary[accepted])
            total_summary.extend(particles_summary)
            c_index.extend(particles_index[accepted]); del(particles_summary); del(particles_index); del(c_shape); del(dis)
            n_alive_particles = n_alive_particles+accepted.sum(); del(accepted)
            print('  '+time.strftime("%H:%M:%S", time.localtime()) + f' Required: {total_n}, Now: {n_alive_particles}', flush=True)

        t_summary = np.array([ii for ii in total_summary]); del(total_summary)
        new_mad = np.nanmedian(np.absolute(t_summary - np.nanmedian(t_summary, axis=0)), axis=0); del(t_summary)

        c_index2 = np.array([ii for ii in c_index]); del(c_index) # just tranform it from a list to an np.array
        self.p_index = c_index2; del(c_index2)

        summary = np.array([ii for ii in c_summary]); del(c_summary)
        dis_now = np.sqrt(np.sum((summary/new_mad)**2, axis=1))
        

        # print(f'Total: {particles_index.shape[0]}', flush=True)
        print(f'Left: {self.p_index.shape[0]}', flush=True)
        print(f'Unique: {np.unique(self.p_index , axis=0).shape[0]} \n', flush=True)

        if details == True:
            np.save(open(file = os.getcwd()+f"/rejection_details/particles_summary.pkl", mode="wb"), summary)
            np.save(open(file = os.getcwd()+f"/rejection_details/p_index.pkl", mode="wb"), self.p_index)
            np.save(open(file = os.getcwd()+f"/rejection_details/new_mad.pkl", mode="wb"), new_mad)
        del(summary); del(dis_now)

        self.rep = rep
        particles = np.array(self.random_IPM_whole_para(total_samples=total_n,
                                                        given_theta=False, given_weight=False, random=False), dtype=object)
        particles_summary = np.array([item for sublist in particles[:, 1] for item in sublist])
        particles_index = np.array([sublist for sublist in particles[:, 0]]); del(particles)
        particles_index = np.repeat(particles_index, rep, axis=0)
        # reshape
        c_shape = particles_summary.shape
        particles_summary = particles_summary.reshape((c_shape[0], c_shape[1]*c_shape[2]))

        dis_now = np.sqrt(np.sum((particles_summary/new_mad)**2, axis=1))
        new_threshold = np.nanquantile(dis_now, axis=0, q=quantile_for_final) 
        
        # accept or reject based on the thershold computed from the previous step.
        accepted = dis_now <= new_threshold
        if details == True:
            np.save(open(file = os.getcwd()+f"/rejection_details/particles_summary_last.pkl", mode="wb"), particles_summary)
            np.save(open(file = os.getcwd()+f"/rejection_details/particles_index_last.pkl", mode="wb"), particles_index)

        sample_final = particles_index[accepted] 
        print(f'Left: {sample_final.shape[0]}', flush=True)
        print(f'Unique: {np.unique(sample_final, axis=0).shape[0]} \n', flush=True)
        if details == True:
            np.save(open(file = os.getcwd()+f"/rejection_details/sample_final.pkl", mode="wb"), sample_final)

        return sample_final






    def smc_rejection(self, index, weight, total_n, rep, quantile_for_final, details=True):

        self.p_index = index
        self.weight = weight 
        self.rep = rep
        particles = np.array(self.random_IPM_whole_para(total_samples=total_n,
                                                        given_theta=True, given_weight=True, random=False), dtype=object)
        particles_summary = np.array([item for sublist in particles[:, 1] for item in sublist])
        particles_index = np.array([sublist for sublist in particles[:, 0]]); del(particles)
        particles_index = np.repeat(particles_index, rep, axis=0)
        # reshape
        c_shape = particles_summary.shape
        particles_summary = particles_summary.reshape((c_shape[0], c_shape[1]*c_shape[2]))

        new_mad = np.nanmedian(np.absolute(particles_summary - np.nanmedian(particles_summary, axis=0)), axis=0)
        dis_now = np.sqrt(np.sum((particles_summary/new_mad)**2, axis=1))
        new_threshold = np.nanquantile(dis_now, axis=0, q=quantile_for_final) 
        
        # accept or reject based on the thershold computed from the previous step.
        accepted = dis_now <= new_threshold
        if details == True:
            np.save(open(file = os.getcwd()+f"/rejection_details/particles_summary_last.pkl", mode="wb"), particles_summary)
            np.save(open(file = os.getcwd()+f"/rejection_details/particles_index_last.pkl", mode="wb"), particles_index)
            np.save(open(file = os.getcwd()+f"/rejection_details/new_threshold.pkl", mode="wb"), new_threshold)
            np.save(open(file = os.getcwd()+f"/rejection_details/dis_now.pkl", mode="wb"), dis_now)

        sample_final = particles_index[accepted] 
        print(f'Left: {sample_final.shape[0]}', flush=True)
        print(f'Unique: {np.unique(sample_final, axis=0).shape[0]} \n', flush=True)
        if details == True:
            np.save(open(file = os.getcwd()+f"/rejection_details/sample_final.pkl", mode="wb"), sample_final)

        return sample_final












    def prediction_para(self, total_samples, abc=False):

        result = []
        n_actor = multiprocessing.cpu_count()-1
        simulators = [para_ipmmcmc_whole.remote(self) for _ in range(n_actor)]
        if abc==False:
            for i in np.arange(0, total_samples, n_actor):
                result.extend(ray.get([s.random_prediction.remote() for s in simulators]))
        else:
            for i in np.arange(0, total_samples, n_actor):
                result.extend(ray.get([s.abc_prediction.remote() for s in simulators]))      

        return result

    def random_prediction(self):

        index = np.random.randint(low=0, high=[ci_niter(self.samples_sur[0].shape[0]),
                                               ci_niter(self.samples_grw_f[0].shape[0]),
                                               ci_niter(self.samples_grw_nf[0].shape[0]),
                                               ci_niter(self.samples_fec[0].shape[0]),
                                               ci_niter(self.samples_flow[0].shape[0])], size=5)
        

        return self.prediction_givenindex(index)


    def abc_prediction(self):
        # randomly select a particle from the previous step.
        index = self.abc_index[np.random.choice(self.abc_index.shape[0], p=self.weight)].copy()
        return self.prediction_givenindex(index) 
     

    def prediction_givenindex(self, index):
        # sur
        for var, var_samples in zip(self.hmc_helper_sur.current_state, self.samples_sur):
            var.assign(var_samples[index[0]])
        # grw_f
        for var, var_samples in zip(self.hmc_helper_grw_f.current_state, self.samples_grw_f):
            var.assign(var_samples[index[1]])
        # grw_nf
        for var, var_samples in zip(self.hmc_helper_grw_nf.current_state, self.samples_grw_nf):
            var.assign(var_samples[index[2]])
        # fec
        for var, var_samples in zip(self.hmc_helper_fec.current_state, self.samples_fec):
            var.assign(var_samples[index[3]])
        # flow
        for var, var_samples in zip(self.hmc_helper_flow.current_state, self.samples_flow):
            var.assign(var_samples[index[4]])

                

        self.m_sur_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/sur/{index[0]}", mode="rb"))
        self.m_grw_f_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/grw_f/{index[1]}", mode="rb")) 
        self.m_grw_nf_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/grw_nf/{index[2]}", mode="rb")) 
        self.m_fec_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/fec/{index[3]}", mode="rb")) 
        self.m_flow_poi_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/flow/{index[4]}", mode="rb")) 
        
        
        models_now2 = {
            "m_sur": self.m_sur_new ,
            "m_grw_f": self.m_grw_f_new,
            "m_grw_nf": self.m_grw_nf_new ,
            "m_fec": self.m_fec_new,
            "m_flow_poi": self.m_flow_poi_new,
            "alpha": self.alpha,
            "beta": self.beta,
            "recruit_p": self.recruit_p
        }

        s = []; n_all = []; n1_all = []; n2_all = []; nb_all = []; nnb_all = []; nf_all = []; sidata = []
        for _ in range(1):
            c_s = []; n_size = []; n1_size = []; n2_size = []; d = []

            models_now2['alpha'] = self.recru_full['alpha'][0]
            models_now2['beta'] = self.recru_full['beta'][0]
            models_now2['recruit_p'] = self.recru_full['recruit_p'][0]

            data_simu = IBM_1step_cashe(zt=self.z0[0], age=self.z0[1], models=models_now2, env=True, 
                                        tmax=self.tmax_list[0], tmin=self.tmin_list[0], precip=self.precip_list[0])
            if (data_simu.shape[0] == 0) or (data_simu.shape[0] == 1):
                return (index, [0, 0, 0, 0, 0, 0, 0, 0], [[np.repeat(np.inf, 5) for _ in range(len(self.popu_all))]])

            current_s = list_comparisons_interested(exp_1step_data=self.popu_all[0], simu_1step_data=data_simu)

            d.append(data_simu)
            c_s.append(current_s)
            n_new1 = np.sum(data_simu["surv"] == 1)
            n_new2 = np.sum(np.isnan(data_simu["size"])) 
            n_size.append(n_new1+n_new2); n1_size.append(n_new1); n2_size.append(n_new2) 
            n_b = np.sum(data_simu.fec == 1) 
            n_nb = np.sum(data_simu.fec == 0)
            n_f = np.nansum(data_simu.flow)

            for t in range(1, len(self.popu_all)):
                if hasattr(self, 'past_recru') and (t > 4):
                    if self.past_recru == True:
                        past_recru_t = np.random.choice(5)
                        models_now2['alpha'] = self.recru_full['alpha'][past_recru_t]
                        models_now2['beta'] = self.recru_full['beta'][past_recru_t]
                        models_now2['recruit_p'] = self.recru_full['recruit_p'][past_recru_t]
                else:
                    models_now2['alpha'] = self.recru_full['alpha'][t]
                    models_now2['beta'] = self.recru_full['beta'][t]
                    models_now2['recruit_p'] = self.recru_full['recruit_p'][t]

                if self.true_z_allsteps == True:
                    z = popu_structure_noid_current(self.popu_all[t])
                    data_simu = IBM_1step_cashe(zt=z[0], age=z[1], models=models_now2, env=True, 
                                                tmax=self.tmax_list[t], tmin=self.tmin_list[t], precip=self.precip_list[t])
                else:
                    z = popu_structure_noid(data_simu) #popu_structure_noid(self.popu_all[t-1])#
                    data_simu = IBM_1step_cashe(zt=z[0], age=z[1], models=models_now2, env=True, 
                                                tmax=self.tmax_list[t], tmin=self.tmin_list[t], precip=self.precip_list[t])

                if (data_simu.shape[0] == 0) or (data_simu.shape[0] == 1):
                    return (index, [0, 0, 0, 0, 0, 0, 0, 0], [[np.repeat(np.inf, 5) for _ in range(len(self.popu_all))]])
                
                try:
                    current_s = list_comparisons_interested(exp_1step_data=self.popu_all[t], simu_1step_data=data_simu)
                except Exception as e:
                    now = datetime.now().strftime("%H_%M_%S")
                    pickle.dump(z , open(file = os.getcwd() + f"/error/z{t}" + now + ".pkl", mode="wb"))
                    pickle.dump(data_simu , open(file = os.getcwd() + "/error/data_simu" + now + ".pkl", mode="wb"))
                    pickle.dump(index, open(file = os.getcwd() + "/error/index" + now + ".pkl", mode="wb"))
                    print(f'Error found at t={t}', flush=True)
                    sys.exit(e)     

                d.append(data_simu)                
                c_s.append(current_s)
                n_new1 = np.sum(data_simu["surv"] == 1)
                n_new2 = np.sum(np.isnan(data_simu["size"])) 
                n_size.append(n_new1+n_new2); n1_size.append(n_new1); n2_size.append(n_new2)
                n_b = np.sum(data_simu.fec == 1) 
                n_nb = np.sum(data_simu.fec == 0)
                n_f = np.nansum(data_simu.flow) 

            s.append(c_s)
            n_all.append(n_size)
            n1_all.append(n1_size)
            n2_all.append(n2_size)
            nb_all.append(n_b)
            nnb_all.append(n_nb)
            nf_all.append(n_f)
            sidata.append(d)

        return (index, n1_all, n2_all, n_all, s, nb_all, nnb_all, nf_all, sidata)





    def return_para_given_index(self, index, modelonly=False):
        # sur
        for var, var_samples in zip(self.hmc_helper_sur.current_state, self.samples_sur):
            var.assign(var_samples[index[0]])
        # grw_f
        for var, var_samples in zip(self.hmc_helper_grw_f.current_state, self.samples_grw_f):
            var.assign(var_samples[index[1]])
        # grw_nf
        for var, var_samples in zip(self.hmc_helper_grw_nf.current_state, self.samples_grw_nf):
            var.assign(var_samples[index[2]])
        # fec
        for var, var_samples in zip(self.hmc_helper_fec.current_state, self.samples_fec):
            var.assign(var_samples[index[3]])
        # flow
        for var, var_samples in zip(self.hmc_helper_flow.current_state, self.samples_flow):
            var.assign(var_samples[index[4]])

        models_now2 = {
            "m_sur": self.m_sur_new ,
            "m_grw_f": self.m_grw_f_new,
            "m_grw_nf": self.m_grw_nf_new ,
            "m_fec": self.m_fec_new,
            "m_flow_poi": self.m_flow_poi_new,
            "alpha": self.alpha,
            "beta": self.beta,
            "recruit_p": self.recruit_p
        }
        if modelonly == True:
            return models_now2 
        theta = extract_parameters_whole(models_now2)
        return theta






@ray.remote
class para_ipmmcmc_whole():
    def __init__(self, ipmmcmc_whole):
        # m is an object belonging to ipmmcmc_whole
        self.ipmmcmc_whole = ipmmcmc_whole
    
    def random_IPM_whole(self):
        return self.ipmmcmc_whole.random_IPM_whole()

    def random_IPM_ABC(self):
        return self.ipmmcmc_whole.random_IPM_ABC()

    def not_random_IPM_ABC(self, i):
        return self.ipmmcmc_whole.not_random_IPM_ABC(i)

    def random_IPM_ABC_weight(self):
        return self.ipmmcmc_whole.random_IPM_ABC_weight()

    def not_random_IPM_ABC_weight(self):
        return self.ipmmcmc_whole.not_random_IPM_ABC_weight()

    def random_prediction(self):
        return self.ipmmcmc_whole.random_prediction()

    def abc_prediction(self):
        return self.ipmmcmc_whole.abc_prediction()



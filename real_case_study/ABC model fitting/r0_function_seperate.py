from r0_function_all import *

########################################################################################################################################################################################################
# when years are fitted separately
# define a naming vectors containing all the parameters' names 
para_names = ['sur_lsize', 'sur_lage', 'sur_kvar',
              'grwf_lsize', 'grwf_lage', 'grwf_kvar', 'grwf_lvar',
              'grwnf_lsize', 'grwnf_lage', 'grwnf_kvar', 'grwnf_lvar',
              'fec_lsize', 'fec_lage', 'fec_kvar',
              'flowp_lsize', 'flowp_lage', 'flowp_kvar',
              'alpha', 'beta', 'recruit_p']

# extract the values of parameters for a given set of models.
def extract_parameters(models):
    theta = pd.DataFrame(0, range(1), para_names)
    theta['alpha'] = models['alpha']
    theta['beta'] = models['beta']
    theta['recruit_p'] = models['recruit_p']

    theta['sur_lsize'], theta['sur_lage'] = np.array(models['m_sur'].kernel.lengthscales); theta['sur_kvar'] = np.array(models['m_sur'].kernel.variance)
    theta['grwf_lsize'], theta['grwf_lage'] = np.array(models['m_grw_f'].kernel.lengthscales); theta['grwf_kvar'] = np.array(models['m_grw_f'].kernel.variance); theta['grwf_lvar'] = np.array(models['m_grw_f'].likelihood.variance) 
    theta['grwnf_lsize'], theta['grwnf_lage'] = np.array(models['m_grw_nf'].kernel.lengthscales); theta['grwnf_kvar'] = np.array(models['m_grw_nf'].kernel.variance); theta['grwnf_lvar'] = np.array(models['m_grw_nf'].likelihood.variance) 

    theta['fec_lsize'], theta['fec_lage'] = np.array(models['m_fec'].kernel.lengthscales); theta['fec_kvar'] = np.array(models['m_fec'].kernel.variance)
    theta['flowp_lsize'], theta['flowp_lage'] = np.array(models['m_flow_poi'].kernel.lengthscales); theta['flowp_kvar'] = np.array(models['m_flow_poi'].kernel.variance)

    return theta

def baseline_model_fit(data_path="control_1011.csv", print_mode = 'OFF', NB=False):
    # data loading.
    if type(data_path) == pd.core.frame.DataFrame:
        data_ori = data_path.copy()
        data_ori = data_ori.sort_values(by=['size'])
        data_ori = data_ori.reset_index(drop=True)

    else:
        assert False, '\n data_path should be a ps.DataFrame !'


    ############################################################################################################################################################# 
    # datasets constructing

    # 1 - surv
    index_sur = np.logical_not(np.isnan(data_ori["surv"]))
    size_sur = data_ori["size"][index_sur].to_numpy()
    surv_sur = data_ori["surv"][index_sur].to_numpy()
    age_sur = data_ori["age"][index_sur].to_numpy()

    # 2 - growth
    # i - if we conserding "fec" effect
    # data for breeders (fec == 1)
    index_grw = np.logical_not(np.isnan(data_ori["sizeNext"])) & np.logical_not(np.isnan(data_ori["size"]))
    fec_grw = data_ori["fec"][index_grw]

    size_grw_f = data_ori["size"][index_grw][fec_grw == 1].to_numpy()
    sizeNext_grw_f = data_ori["sizeNext"][index_grw][fec_grw == 1].to_numpy()
    age_grw_f = data_ori["age"][index_grw][fec_grw == 1].to_numpy()

    # data for breeders (fec == 0)
    size_grw_nf = data_ori["size"][index_grw][fec_grw == 0].to_numpy()
    sizeNext_grw_nf = data_ori["sizeNext"][index_grw][fec_grw == 0].to_numpy()
    age_grw_nf = data_ori["age"][index_grw][fec_grw == 0].to_numpy()

    # 3 - fec
    index_fec = np.logical_not(np.isnan(data_ori["fec"]))
    size_fec = data_ori["size"][index_fec].to_numpy()
    fec_fec = data_ori["fec"][index_fec].to_numpy()
    age_fec = data_ori["age"][index_fec].to_numpy()
    random_index = np.random.choice(500, 50)

    # 4 - number of flowering stalks
    index_flow = data_ori["fec"] == 1
    size_flow = data_ori["size"][index_flow].to_numpy()
    flow_flow = data_ori["flow"][index_flow].to_numpy()
    age_flow = data_ori["age"][index_flow].to_numpy()

    # 5 - recruit size (indp with the parent size)
    index_recr = np.isnan(data_ori["size"])
    

    if type(data_path) == pd.core.frame.DataFrame:
        # log have already taken for these simulated datasets

        # 1 - surv
        X_sur = np.concatenate((size_sur[:, None], age_sur[:, None]), axis=1)
        Y_sur = surv_sur[:, None]

        # 2 - growth
        X_grw_f = np.concatenate((size_grw_f[:, None], age_grw_f[:, None]), axis=1)
        Y_grw_f = sizeNext_grw_f[:, None]
        X_grw_nf = np.concatenate((size_grw_nf[:, None], age_grw_nf[:, None]), axis=1)
        Y_grw_nf = sizeNext_grw_nf[:, None]

        # 3 - fec
        X_fec = np.concatenate((size_fec[:, None], age_fec[:, None]), axis=1)
        Y_fec = fec_fec[:, None]

        # 4 - number of flowering stalks
        X_flow = np.concatenate((size_flow[:, None], age_flow[:, None]), axis=1)
        Y_flow = flow_flow[:, None] - 1
        sizeNext_recr = data_ori["sizeNext"][index_recr].to_numpy()

    else: #type(data_path) == str:
        assert False, '\n data_path should be a ps.DataFrame !'

    
    ############################################################################################################################################################# 
    # model fitting
    # 1 - survive 
    l_sur = gpflow.likelihoods.Bernoulli()
    k_sur = gpflow.kernels.RBF(lengthscales=np.array([1.4,0.7]))

    m_sur1 = gpflow.models.VGP((X_sur, Y_sur), kernel=k_sur, likelihood=l_sur)

    opt_sur = gpflow.optimizers.Scipy()
    opt_sur.minimize(m_sur1.training_loss, variables=m_sur1.trainable_variables, options=dict(maxiter=10000))

    # 2 - growth: if we conserding "fec" effect
    # growth model fitting for breeders (fec == 1)

    k_grw_f = gpflow.kernels.RBF(lengthscales=np.array([1,1000]))

    m_grw_f = gpflow.models.GPR(data=(X_grw_f, Y_grw_f), 
                                kernel=k_grw_f, mean_function=None)

    opt_grw_f = gpflow.optimizers.Scipy()
    opt_grw_f.minimize(m_grw_f.training_loss, variables=m_grw_f.trainable_variables, options=dict(maxiter=10000))


    # growth model fitting for the remainings (fec == 0)
    k_grw_nf = gpflow.kernels.RBF(lengthscales=np.array([1,1]))

    m_grw_nf = gpflow.models.GPR(data=(X_grw_nf, Y_grw_nf), 
                                kernel=k_grw_nf, mean_function=None)

    opt_grw_nf = gpflow.optimizers.Scipy()
    opt_grw_nf.minimize(m_grw_nf.training_loss, variables=m_grw_nf.trainable_variables, options=dict(maxiter=10000))


    # 3 - fec
    l_fec = gpflow.likelihoods.Bernoulli()
    k_fec = gpflow.kernels.RBF(lengthscales=np.array([1,990]), variance=6)
    m_fec = gpflow.models.VGP((X_fec, Y_fec), kernel=k_fec, likelihood=l_fec)

    opt_fec = gpflow.optimizers.Scipy()
    opt_fec.minimize(m_fec.training_loss, variables=m_fec.trainable_variables, options=dict(maxiter=10000))



    # 4 - number of flowering stalks 
    # (i) Poisson
    l_flow_poi = gpflow.likelihoods.Poisson()
    k_flow_poi = gpflow.kernels.RBF(lengthscales=np.array([1,1]))

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


    if print_mode == 'ON':
        print(m_sur1.training_loss())
        gpflow.utilities.print_summary(m_sur1, fmt="notebook")
        print(m_grw_f.training_loss())
        gpflow.utilities.print_summary(m_grw_f, fmt="notebook")
        print(m_grw_nf.training_loss())
        gpflow.utilities.print_summary(m_grw_nf, fmt="notebook")
        print(m_fec.training_loss())
        gpflow.utilities.print_summary(m_fec, fmt="notebook")
        print(m_flow_poi.training_loss())
        gpflow.utilities.print_summary(m_flow_poi, fmt="notebook")
        print(m_flow_nb.training_loss())
        gpflow.utilities.print_summary(m_flow_nb, fmt="notebook")


    models_true = {
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

    return models_true



class ipmmcmc():
    def __init__(self, popu_dataset, alpha, beta, recruit_p):
        self.popu_dataset = popu_dataset
        self.alpha = alpha
        self.beta = beta
        self.recruit_p = recruit_p
        self.z = popu_structure_noid_current(self.popu_dataset)
        # np.random.seed(11111)
        # self.popu_dataset_true = IBM_1step(zt=self.z[0], age=self.z[1], models=self.models_true)

    def random_IPM_simple_para(self, total_samples):
        # if adaptive=True, then the algorithm will not accpect or reject particles based on the input threshold
        result = []
        n_actor = multiprocessing.cpu_count()-1
        simulators = [para_ipmmcmc.remote(self) for _ in range(n_actor)]
        for i in np.arange(0, total_samples, n_actor):
            result.extend(ray.get([s.random_IPM_simple.remote() for s in simulators]))

        return result

    def random_IPM_simple(self):
        index = np.random.randint(low=0, high=self.num_samples, size=5)
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

        s = []

        for _ in range(self.rep):
            data_simu=IBM_1step(zt=self.z[0], age=self.z[1], env=False, models=models_now2)
            current_s = list_comparisons_interested(exp_1step_data=self.popu_dataset, simu_1step_data=data_simu)
            s.append(current_s)

        return (index, s)            

@ray.remote
class para_ipmmcmc():
    def __init__(self, para_ipmmcmc):
        # m is an object belonging to ipmmcmc
        self.para_ipmmcmc = para_ipmmcmc
    
    def random_IPM(self):
        return self.para_ipmmcmc.random_IPM()

    def random_IPM_simple(self):
        return self.para_ipmmcmc.random_IPM_simple()





def ipmmcmc_1year(popu_dataset, alpha, beta, recruit_p, 
                  samples_grw_nf, samples_fec, samples_flow, samples_grw_f, samples_sur,
                  rep, total_samples):

    # if adaptive=True, then the algorithm will not accpect or reject particles based on the input threshold
                  
    single_year = ipmmcmc(popu_dataset=popu_dataset, alpha=alpha, beta=beta, recruit_p=recruit_p)

    # Loading MCMC samples
    # HMC for the other vital rates.
    print('\n\n\n' + 'Processing: HMC' + '\n\n\n')
    single_year.num_samples = ci_niter(samples_grw_nf[0].shape[0])

    # Grw_nf
    single_year.m_grw_nf_new = gpflow.models.GPR(data=(XY_grw_nf_compu(popu_dataset)), 
                                                 kernel=gpflow.kernels.RBF(lengthscales=np.array([1,1])), 
                                                 mean_function=None)

    # we add priors to the hyperparameters.
    single_year.m_grw_nf_new.kernel.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
    single_year.m_grw_nf_new.likelihood.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
    single_year.m_grw_nf_new.kernel.lengthscales.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
    single_year.hmc_helper_grw_nf = gpflow.optimizers.SamplingHelper(
        single_year.m_grw_nf_new.log_posterior_density, single_year.m_grw_nf_new.trainable_parameters
    )

    single_year.samples_grw_nf = samples_grw_nf

    # Fec
    single_year.m_fec_new = gpflow.models.GPMC(data=(XY_fec_compu(popu_dataset)), 
                                kernel=gpflow.kernels.RBF(lengthscales=np.array([1,1])) , 
                                likelihood=gpflow.likelihoods.Bernoulli())

    single_year.m_fec_new.kernel.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
    single_year.m_fec_new.kernel.lengthscales.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))

    single_year.hmc_helper_fec = gpflow.optimizers.SamplingHelper(
        single_year.m_fec_new.log_posterior_density, single_year.m_fec_new.trainable_parameters
    )

    single_year.samples_fec = samples_fec

    # Flow poi
    single_year.m_flow_poi_new = gpflow.models.GPMC(data=(XY_flow_compu(popu_dataset)), 
                                        kernel=gpflow.kernels.RBF(lengthscales=np.array([1,1])), 
                                        likelihood=gpflow.likelihoods.Poisson())

    single_year.m_flow_poi_new.kernel.lengthscales.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
    single_year.m_flow_poi_new.kernel.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))

    single_year.hmc_helper_flow = gpflow.optimizers.SamplingHelper(
        single_year.m_flow_poi_new.log_posterior_density, single_year.m_flow_poi_new.trainable_parameters
    )

    single_year.samples_flow = samples_flow


    # Grw_f
    single_year.m_grw_f_new = gpflow.models.GPR(data=(XY_grw_f_compu(popu_dataset)), 
                                    kernel= gpflow.kernels.RBF(lengthscales=np.array([1,1])), mean_function=None)

    single_year.m_grw_f_new.kernel.lengthscales.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
    single_year.m_grw_f_new.kernel.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
    single_year.m_grw_f_new.likelihood.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))

    single_year.hmc_helper_grw_f = gpflow.optimizers.SamplingHelper(
        single_year.m_grw_f_new.log_posterior_density, single_year.m_grw_f_new.trainable_parameters
    )

    single_year.samples_grw_f = samples_grw_f

    # Sur
    single_year.m_sur_new = gpflow.models.GPMC(data=(XY_sur_compu(popu_dataset)), 
                                kernel=gpflow.kernels.RBF(lengthscales=np.array([1,1])) , 
                                likelihood=gpflow.likelihoods.Bernoulli())

    single_year.m_sur_new.kernel.lengthscales.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))
    single_year.m_sur_new.kernel.variance.prior = tfd.Normal(loc=f64(0.), scale=f64(100.))

    single_year.hmc_helper_sur = gpflow.optimizers.SamplingHelper(
        single_year.m_sur_new.log_posterior_density, single_year.m_sur_new.trainable_parameters
    )

    single_year.samples_sur = samples_sur
    
    print('\n\n\n' + 'Simulation starts!' + '\n\n\n')

    single_year.rep = rep
    #results = np.array(single_year.random_IPM_para(total_samples=total_samples), dtype=object)
    results = np.array(single_year.random_IPM_simple_para(total_samples=total_samples), dtype=object)

    return results        





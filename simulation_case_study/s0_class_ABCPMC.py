import pandas as pd
import numpy as np
import multiprocessing
import ray
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.ci_utils import reduce_in_tests as ci_niter
from tensorflow_probability import distributions as tfd
import pickle
import os
from s0_fun_IBMs import IBM_1step_gp_cache, IBM_1step_gp_same_cache
from s0_fun_ss import list_comparisons_interested
import sys
import time
f64 = gpflow.utilities.to_default_float

# Notice that, some code involve attribute like 'grw_setting == "sep"', which is not a part of this project.


########################################################################################################################################################################################################
# when years are fitted altogether

# define a naming vectors containing all the parameters' names 
para_names_sep = ['sur_lsize', 'sur_kvar',
                    'grwf_lsize', 'grwf_kvar', 'grwf_lvar',
                    'grwnf_lsize', 'grwnf_kvar', 'grwnf_lvar',
                    'fec_lsize', 'fec_kvar',
                    'flowp_lsize', 'flowp_kvar',
                    'alpha', 'beta', 'recruit_p']

para_names_nonsep = ['sur_lsize', 'sur_kvar',
                    'grw_lsize', 'grw_kvar', 'grw_lvar',
                    'fec_lsize', 'fec_kvar',
                    'flowp_lsize', 'flowp_kvar',
                    'alpha', 'beta', 'recruit_p']

def extract_parameters_whole(models, grw_setting):
    if grw_setting == 'sep':
        theta = pd.DataFrame(0, range(1), para_names_sep)
        theta['alpha'] = models['alpha']
        theta['beta'] = models['beta']
        theta['recruit_p'] = models['recruit_p']

        theta['sur_lsize'] = np.array(models['m_sur'].kernel.lengthscales)
        theta['sur_kvar'] = np.array(models['m_sur'].kernel.variance)

        theta['grwf_lsize'] = np.array(models['m_grw_f'].kernel.lengthscales)
        theta['grwf_kvar'] = np.array(models['m_grw_f'].kernel.variance); theta['grwf_lvar'] = np.array(models['m_grw_f'].likelihood.variance) 
        
        theta['grwnf_lsize'] = np.array(models['m_grw_nf'].kernel.lengthscales)
        theta['grwnf_kvar'] = np.array(models['m_grw_nf'].kernel.variance); theta['grwnf_lvar'] = np.array(models['m_grw_nf'].likelihood.variance) 

        theta['fec_lsize'] = np.array(models['m_fec'].kernel.lengthscales)
        theta['fec_kvar'] = np.array(models['m_fec'].kernel.variance)

        theta['flowp_lsize'] = np.array(models['m_flow_poi'].kernel.lengthscales)
        theta['flowp_kvar'] = np.array(models['m_flow_poi'].kernel.variance)
    elif grw_setting == 'nonsep':
        theta = pd.DataFrame(0, range(1), para_names_sep)
        theta['alpha'] = models['alpha']
        theta['beta'] = models['beta']
        theta['recruit_p'] = models['recruit_p']

        theta['sur_lsize'] = np.array(models['m_sur'].kernel.lengthscales)
        theta['sur_kvar'] = np.array(models['m_sur'].kernel.variance)

        theta['grw_lsize'] = np.array(models['m_grw_f'].kernel.lengthscales)
        theta['grw_kvar'] = np.array(models['m_grw_f'].kernel.variance); theta['grwf_lvar'] = np.array(models['m_grw_f'].likelihood.variance) 

        theta['fec_lsize'] = np.array(models['m_fec'].kernel.lengthscales)
        theta['fec_kvar'] = np.array(models['m_fec'].kernel.variance)

        theta['flowp_lsize'] = np.array(models['m_flow_poi'].kernel.lengthscales)
        theta['flowp_kvar'] = np.array(models['m_flow_poi'].kernel.variance)
    
    else:
        assert False, '\n Unkown grw_setting' 
    return theta


# our main function and class.
#  IPM_whole is standing for the big IPM build for all years 
class ipmmcmc_whole():
    def __init__(self, popu_data, z0, grw_setting, truedata_style, alpha, beta, recruit_p):
        # z0: the initial population structure
        self.popu_data = popu_data
        self.grw_setting = grw_setting
        self.truedata_style = truedata_style
        self.alpha = alpha
        self.beta = beta
        self.recruit_p = recruit_p
        self.z0 = z0

    # build IPM_whole by given random indeces for each vital rate's MCMC sample (in parallel computing)
    def random_IPM_whole_para(self, total_samples, given_theta=False, given_weight=False, random=True):

        result = []
        n_actor = multiprocessing.cpu_count()-2
        simulators = [para_ipmmcmc_whole.remote(self) for _ in range(n_actor)]
        if given_theta==False and given_weight==False and random==True:
            for i in np.arange(0, total_samples, n_actor):
                result.extend(ray.get([s.random_IPM_whole.remote() for s in simulators]))
        elif given_theta==True and given_weight==False and random==True:
            for i in np.arange(0, total_samples, n_actor):
                result.extend(ray.get([s.random_IPM_ABC.remote() for s in simulators])) 
        elif given_theta==True and given_weight==True and random==True:
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
        if self.grw_setting == "sep":

            index = np.random.randint(low=0, high=[ci_niter(self.samples_sur[0].shape[0]),
                                                ci_niter(self.samples_grw_f[0].shape[0]),
                                                ci_niter(self.samples_grw_nf[0].shape[0]),
                                                ci_niter(self.samples_fec[0].shape[0]),
                                                ci_niter(self.samples_flow[0].shape[0])], size=5)
        else: 

            index = np.random.randint(low=0, high=[ci_niter(self.samples_sur[0].shape[0]),
                                                ci_niter(self.samples_grw[0].shape[0]),
                                                ci_niter(self.samples_fec[0].shape[0]),
                                                ci_niter(self.samples_flow[0].shape[0])], size=4)

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

        if self.grw_setting == "sep":
            index[np.random.randint(0, 5)] = np.random.randint(0, 5000) 
        else:
            index[np.random.randint(0, 4)] = np.random.randint(0, 5000)  
        
        return self.IPM_whole_givenindex(index) 


    def new_model(self, index):
        if self.grw_setting == 'sep':
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
            self.m_sur_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + self.truedata_style + f"/sur/{index[0]}", mode="rb"))
            self.m_grw_f_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + self.truedata_style + f"/grw_f/{index[1]}", mode="rb")) 
            self.m_grw_nf_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + self.truedata_style + f"/grw_nf/{index[2]}", mode="rb")) 
            self.m_fec_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + self.truedata_style + f"/fec/{index[3]}", mode="rb")) 
            self.m_flow_poi_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + self.truedata_style + f"/flow/{index[4]}", mode="rb")) 

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

        elif self.grw_setting == 'nonsep':
            # sur
            for var, var_samples in zip(self.hmc_helper_sur.current_state, self.samples_sur):
                var.assign(var_samples[index[0]])
            # grw
            for var, var_samples in zip(self.hmc_helper_grw.current_state, self.samples_grw):
                var.assign(var_samples[index[1]])
            # fec
            for var, var_samples in zip(self.hmc_helper_fec.current_state, self.samples_fec):
                var.assign(var_samples[index[2]])
            # flow
            for var, var_samples in zip(self.hmc_helper_flow.current_state, self.samples_flow):
                var.assign(var_samples[index[3]])

            # cache (used for cholesky decomposition) for sampled MCMC are calcualted and stored
            #  the index for the MCMC samples are consistent with the corresponding stored files' name
            self.m_sur_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + self.truedata_style + f"/sur/{index[0]}", mode="rb"))
            self.m_grw_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + self.truedata_style + f"/grw/{index[1]}", mode="rb")) 
            self.m_fec_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + self.truedata_style + f"/fec/{index[2]}", mode="rb")) 
            self.m_flow_poi_new.cache = pickle.load(open(file = os.getcwd()+f"/true_popu/mcmc/Lm/" + self.truedata_style + f"/flow/{index[3]}", mode="rb")) 

            models_now2 = {
                "m_sur": self.m_sur_new ,
                "m_grw": self.m_grw_new,
                "m_fec": self.m_fec_new,
                "m_flow_poi": self.m_flow_poi_new,
                "alpha": self.alpha,
                "beta": self.beta,
                "recruit_p": self.recruit_p
            }

        return models_now2 


    def IPM_whole_givenindex(self, index):
        models_now2 = self.new_model(index)
        s = []
        for _ in range(self.rep):
            # keep c_s = [] and c_s.append(current_s) is just for keep structure of data unchanged
            #  when there is mutiple years needed to be simulated. (for t in range(1, len(self.popu_0toT)):)
            c_s = []

            if self.grw_setting == 'sep':
                data_simu = IBM_1step_gp_cache(zt=self.z0[0], age=self.z0[1], models=models_now2)
            elif self.grw_setting == 'nonsep':
                data_simu = IBM_1step_gp_same_cache(zt=self.z0[0], age=self.z0[1], models=models_now2)
            
            current_s = list_comparisons_interested(exp_1step_data=self.popu_data, simu_1step_data=data_simu, truedata_style=self.truedata_style)
            c_s.append(current_s)

            s.append(c_s)

        return (index, s)
    
    # a function used for cache information ONLY
    def GPmodels_givenindex_cache(self, index):
        print('This is a function used for cache information ONLY, please DO NOT use it for doing simulations.')
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
        # grw
        for var, var_samples in zip(self.hmc_helper_grw.current_state, self.samples_grw):
            var.assign(var_samples[index[5]])

        models_now2 = {
            "m_sur": self.m_sur_new,
            "m_grw_f": self.m_grw_f_new,
            "m_grw_nf": self.m_grw_nf_new,
            "m_grw": self.m_grw_new,
            "m_fec": self.m_fec_new,
            "m_flow_poi": self.m_flow_poi_new,
            "alpha": self.alpha,
            "beta": self.beta,
            "recruit_p": self.recruit_p
        }



    def ABC_SMC(self, quantiles, n_particles, details=True,
                algo_continue=False, hist_mad=None, hist_threshold=None, hist_index=None, hist_weight=None):
        # algo_continue=True has NOT been tested yet.

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
                np.save(open(file = os.getcwd()+"/ABC_details/"+self.truedata_style+self.grw_setting+"/p_index_ini.pkl", mode="wb"), particles_index)

            # reshape
            c_shape = particles_summary.shape
            particles_summary = particles_summary.reshape((c_shape[0], c_shape[1]*c_shape[2]))
            self.mad = np.nanmedian(np.absolute(particles_summary - np.nanmedian(particles_summary, axis=0)), axis=0)
            s_mean = np.nanmean(particles_summary, axis=0)
            s_median = np.nanmedian(particles_summary, axis=0)
            dis = np.sqrt(np.sum((particles_summary/self.mad)**2, axis=1))


            # we accept all the particles at c=0 in this version.
            self.threshold = np.nanquantile(dis, axis=0, q=quantiles[0])
            accepted = np.repeat(True, particles_index.shape[0]) 

            self.p_index = particles_index[accepted]
            # equal weights
            self.weight = np.repeat(1/np.sum(accepted), np.sum(accepted))

            print(f'Total: {particles_index.shape[0]}', flush=True)
            print(f'Left: {self.p_index.shape[0]}', flush=True)
            print(f'Unique: {np.unique(self.p_index , axis=0).shape[0]} \n', flush=True)

            if details == True:
                np.save(open(file = os.getcwd()+f"/ABC_details/"+self.truedata_style+self.grw_setting+f"/particles_summary{0}.pkl", mode="wb"), particles_summary)
                np.save(open(file = os.getcwd()+f"/ABC_details/"+self.truedata_style+self.grw_setting+f"/accepted{0}.pkl", mode="wb"), accepted)
                np.save(open(file = os.getcwd()+f"/ABC_details/"+self.truedata_style+self.grw_setting+f"/p_index_{0}.pkl", mode="wb"), self.p_index)
                np.save(open(file = os.getcwd()+f"/ABC_details/"+self.truedata_style+self.grw_setting+f"/threshold{0}.pkl", mode="wb"), self.threshold)
                np.save(open(file = os.getcwd()+f"/ABC_details/"+self.truedata_style+self.grw_setting+f"/weight{0}.pkl", mode="wb"), self.weight)
                np.save(open(file = os.getcwd()+f"/ABC_details/"+self.truedata_style+self.grw_setting+f"/mad{0}.pkl", mode="wb"), self.mad)
                np.save(open(file = os.getcwd()+f"/ABC_details/"+self.truedata_style+self.grw_setting+f"/r_s_mean{0}.pkl", mode="wb"), s_mean)
                np.save(open(file = os.getcwd()+f"/ABC_details/"+self.truedata_style+self.grw_setting+f"/r_s_median{0}.pkl", mode="wb"), s_median)
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
                if self.grw_setting == 'sep':
                    neighbour = np.sum(self.p_index == c_index2[kk], axis=1) >= 4
                elif self.grw_setting == 'nonsep':
                    neighbour = np.sum(self.p_index == c_index2[kk], axis=1) >= 3
                c_weight.append(1/np.sum(self.weight[neighbour]))

            self.weight = c_weight/np.sum(c_weight); del(c_weight)
            self.p_index = c_index2; del(c_index2)
            summary = np.array([ii for ii in c_summary]); del(c_summary)
            dis_now = np.sqrt(np.sum((summary/self.mad)**2, axis=1))
            self.threshold = np.nanquantile(dis_now, axis=0, q=quantiles[c])

            print(f'Left: {self.p_index.shape[0]}', flush=True)
            print(f'Unique: {np.unique(self.p_index , axis=0).shape[0]} \n', flush=True)

            if details == True:
                np.save(open(file = os.getcwd()+f"/ABC_details/"+self.truedata_style+self.grw_setting+f"/particles_summary{c}.pkl", mode="wb"), summary)
                np.save(open(file = os.getcwd()+f"/ABC_details/"+self.truedata_style+self.grw_setting+f"/p_index_{c}.pkl", mode="wb"), self.p_index)
                np.save(open(file = os.getcwd()+f"/ABC_details/"+self.truedata_style+self.grw_setting+f"/threshold{c}.pkl", mode="wb"), self.threshold)
                np.save(open(file = os.getcwd()+f"/ABC_details/"+self.truedata_style+self.grw_setting+f"/weight{c}.pkl", mode="wb"), self.weight)
                np.save(open(file = os.getcwd()+f"/ABC_details/"+self.truedata_style+self.grw_setting+f"/mad{c}.pkl", mode="wb"), self.mad)
                np.save(open(file = os.getcwd()+f"/ABC_details/"+self.truedata_style+self.grw_setting+f"/r_s_mean{c}.pkl", mode="wb"), s_mean)
                np.save(open(file = os.getcwd()+f"/ABC_details/"+self.truedata_style+self.grw_setting+f"/r_s_median{c}.pkl", mode="wb"), s_median)
            del(summary)

        return self.p_index, self.threshold








    def prediction_para(self, total_samples, abc=False):

        result = []
        n_actor = multiprocessing.cpu_count()-2
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
        index = self.abc_index[np.random.choice(self.abc_index.shape[0], p=self.weight)].copy()
        return self.prediction_givenindex(index) 


    def prediction_givenindex(self, index):
        
        models_now2 = self.new_model(index) 

        s = []; n_all = []; n1_all = []; n2_all = []; nb_all = []; nnb_all = []; nf_all = []; sidata = []
        for _ in range(1):
            c_s = []; n_size = []; n1_size = []; n2_size = []; d = []

            if self.grw_setting == 'sep':
                data_simu = IBM_1step_gp_cache(zt=self.z0[0], age=self.z0[1], models=models_now2)
            elif self.grw_setting == 'nonsep':
                data_simu = IBM_1step_gp_same_cache(zt=self.z0[0], age=self.z0[1], models=models_now2)
        
            current_s = list_comparisons_interested(exp_1step_data=self.popu_data, simu_1step_data=data_simu, truedata_style=self.truedata_style)

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

        models_now2 = self.new_model(index)

        if modelonly == True:
            return models_now2 
        theta = extract_parameters_whole(models_now2, grw_setting=self.grw_setting)
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
    


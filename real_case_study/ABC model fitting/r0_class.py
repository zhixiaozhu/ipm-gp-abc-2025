
import pandas as pd
import numpy as np
import multiprocessing
import ray
from r0_function import popu_structure_noid, IBM_1step, popu_structure_noid_current

# ipm_sep is to build individual IPMs for each SINGLE year
class ipm_sep():
    def __init__(self, popu_0toT, alpha_array, beta_array, recruit_p_array, n_train, n_pred):
        # n_train: the number of training datasets
        self.n_train=n_train
        # n_pred: the number of predicted year 
        self.n_pred=n_pred
        self.popu_0toT = popu_0toT
        # starting population structure for the first predicted year year.
        self.z_p = popu_structure_noid(self.popu_0toT[n_train-1])
        # starting population structure for Year 0.
        self.z0 = popu_structure_noid_current(self.popu_0toT[0])

        self.alpha_array = alpha_array
        self.beta_array = beta_array
        self.recruit_p_array = recruit_p_array

    def random_lambda_para(self, total_samples):
        # if adaptive=True, then the algorithm will not accpect or reject particles based on the input threshold
        result = []
        n_actor = multiprocessing.cpu_count()-1
        simulators = [para_ipm_sep.remote(self) for _ in range(n_actor)]
        for i in np.arange(0, total_samples, n_actor):
            result.extend(ray.get([s.random_lambda.remote() for s in simulators]))

        return result

    def random_lambda(self):
        # we are forcasting the next self.n_pred years.
        year_index = np.random.randint(low=0, high=5, size=self.n_pred)
        current_z = self.z_p
        l = []
        n = []

        for i in range(self.n_pred):

            index = np.random.randint(low=0, high=self.num_samples, size=5)
            # sur
            for var, var_samples in zip(self.hmc_helper_sur[year_index[i]].current_state, self.samples_sur[year_index[i]]):
                var.assign(var_samples[index[0]])
            # grw_f
            for var, var_samples in zip(self.hmc_helper_grw_f[year_index[i]].current_state, self.samples_grw_f[year_index[i]]):
                var.assign(var_samples[index[1]])
            # grw_nf
            for var, var_samples in zip(self.hmc_helper_grw_nf[year_index[i]].current_state, self.samples_grw_nf[year_index[i]]):
                var.assign(var_samples[index[2]])
            # fec
            for var, var_samples in zip(self.hmc_helper_fec[year_index[i]].current_state, self.samples_fec[year_index[i]]):
                var.assign(var_samples[index[3]])
            # flow
            for var, var_samples in zip(self.hmc_helper_flow[year_index[i]].current_state, self.samples_flow[year_index[i]]):
                var.assign(var_samples[index[4]])

            models_now2 = {
                "m_sur": self.m_sur_new[year_index[i]],
                "m_grw_f": self.m_grw_f_new[year_index[i]],
                "m_grw_nf": self.m_grw_nf_new[year_index[i]],
                "m_fec": self.m_fec_new[year_index[i]],
                "m_flow_poi": self.m_flow_poi_new[year_index[i]],
                "alpha": self.alpha_array[year_index[i]],
                "beta": self.beta_array[year_index[i]],
                "recruit_p": self.recruit_p_array[year_index[i]]
            }

            data_simu = IBM_1step(zt=current_z[0], age=current_z[1], models=models_now2)
            current_z = popu_structure_noid(data_simu)

            n_new1 = np.sum(data_simu["surv"] == 1)
            n_new2 = np.sum(np.isnan(data_simu["size"])) 
            n_old = np.sum(np.logical_not(np.isnan(data_simu["size"])))

            # growth rate
            l.append((n_new1+n_new2)/n_old)
            # population size
            n.append(n_new1+n_new2)

        # Now, for ABC
        year_index = np.random.randint(low=0, high=5, size=self.n_pred)
        current_z = self.z_p
        l_ABC = []
        n_ABC = []

        for i in range(self.n_pred):
            sample_index = np.random.randint(low=0, high=np.sum(self.accepted_e[year_index[i]]), size=1)
            index = self.index[year_index[i]][self.accepted_e[year_index[i]]][sample_index][0]

            # sur
            for var, var_samples in zip(self.hmc_helper_sur[year_index[i]].current_state, self.samples_sur[year_index[i]]):
                var.assign(var_samples[index[0]])
            # grw_f
            for var, var_samples in zip(self.hmc_helper_grw_f[year_index[i]].current_state, self.samples_grw_f[year_index[i]]):
                var.assign(var_samples[index[1]])
            # grw_nf
            for var, var_samples in zip(self.hmc_helper_grw_nf[year_index[i]].current_state, self.samples_grw_nf[year_index[i]]):
                var.assign(var_samples[index[2]])
            # fec
            for var, var_samples in zip(self.hmc_helper_fec[year_index[i]].current_state, self.samples_fec[year_index[i]]):
                var.assign(var_samples[index[3]])
            # flow
            for var, var_samples in zip(self.hmc_helper_flow[year_index[i]].current_state, self.samples_flow[year_index[i]]):
                var.assign(var_samples[index[4]])

            models_now2 = {
                "m_sur": self.m_sur_new[year_index[i]],
                "m_grw_f": self.m_grw_f_new[year_index[i]],
                "m_grw_nf": self.m_grw_nf_new[year_index[i]],
                "m_fec": self.m_fec_new[year_index[i]],
                "m_flow_poi": self.m_flow_poi_new[year_index[i]],
                "alpha": self.alpha_array[year_index[i]],
                "beta": self.beta_array[year_index[i]],
                "recruit_p": self.recruit_p_array[year_index[i]]
            }

            data_simu = IBM_1step(zt=current_z[0], age=current_z[1], models=models_now2)
            current_z = popu_structure_noid(data_simu)

            n_new1 = np.sum(data_simu["surv"] == 1)
            n_new2 = np.sum(np.isnan(data_simu["size"])) 
            n_old = np.sum(np.logical_not(np.isnan(data_simu["size"])))

            # growth rate
            l_ABC.append((n_new1+n_new2)/n_old)
            # population size
            n_ABC.append(n_new1+n_new2)


        return (l, n, l_ABC, n_ABC, year_index)            


    def random_lambda_full_para(self, total_samples):
        # if adaptive=True, then the algorithm will not accpect or reject particles based on the input threshold
        result = []
        n_actor = multiprocessing.cpu_count()-1
        simulators = [para_ipm_sep.remote(self) for _ in range(n_actor)]
        for i in np.arange(0, total_samples, n_actor):
            result.extend(ray.get([s.random_lambda_full.remote() for s in simulators]))

        return result

    def random_lambda_full(self):
        # we are forcasting the next 5 years.
        year_index = np.concatenate((np.arange(self.n_train), np.random.randint(low=0, high=5, size=self.n_pred)))
        current_z = self.z_p
        l = []
        n = []

        for i in range(self.n_pred+self.n_train):

            index = np.random.randint(low=0, high=self.num_samples, size=5)
            # sur
            for var, var_samples in zip(self.hmc_helper_sur[year_index[i]].current_state, self.samples_sur[year_index[i]]):
                var.assign(var_samples[index[0]])
            # grw_f
            for var, var_samples in zip(self.hmc_helper_grw_f[year_index[i]].current_state, self.samples_grw_f[year_index[i]]):
                var.assign(var_samples[index[1]])
            # grw_nf
            for var, var_samples in zip(self.hmc_helper_grw_nf[year_index[i]].current_state, self.samples_grw_nf[year_index[i]]):
                var.assign(var_samples[index[2]])
            # fec
            for var, var_samples in zip(self.hmc_helper_fec[year_index[i]].current_state, self.samples_fec[year_index[i]]):
                var.assign(var_samples[index[3]])
            # flow
            for var, var_samples in zip(self.hmc_helper_flow[year_index[i]].current_state, self.samples_flow[year_index[i]]):
                var.assign(var_samples[index[4]])

            models_now2 = {
                "m_sur": self.m_sur_new[year_index[i]],
                "m_grw_f": self.m_grw_f_new[year_index[i]],
                "m_grw_nf": self.m_grw_nf_new[year_index[i]],
                "m_fec": self.m_fec_new[year_index[i]],
                "m_flow_poi": self.m_flow_poi_new[year_index[i]],
                "alpha": self.alpha_array[year_index[i]],
                "beta": self.beta_array[year_index[i]],
                "recruit_p": self.recruit_p_array[year_index[i]]
            }

            data_simu = IBM_1step(zt=current_z[0], age=current_z[1], models=models_now2)
            current_z = popu_structure_noid(data_simu)

            n_new1 = np.sum(data_simu["surv"] == 1)
            n_new2 = np.sum(np.isnan(data_simu["size"])) 
            n_old = np.sum(np.logical_not(np.isnan(data_simu["size"])))

            # growth rate
            l.append((n_new1+n_new2)/n_old)
            # population size
            n.append(n_new1+n_new2)

        # Now, for ABC
        year_index = np.concatenate((np.arange(self.n_train), np.random.randint(low=0, high=5, size=self.n_pred)))
        current_z = self.z_p
        l_ABC = []
        n_ABC = []

        for i in range(self.n_pred+self.n_train):
            sample_index = np.random.randint(low=0, high=np.sum(self.accepted_e[year_index[i]]), size=1)
            index = self.index[year_index[i]][self.accepted_e[year_index[i]]][sample_index][0]

            # sur
            for var, var_samples in zip(self.hmc_helper_sur[year_index[i]].current_state, self.samples_sur[year_index[i]]):
                var.assign(var_samples[index[0]])
            # grw_f
            for var, var_samples in zip(self.hmc_helper_grw_f[year_index[i]].current_state, self.samples_grw_f[year_index[i]]):
                var.assign(var_samples[index[1]])
            # grw_nf
            for var, var_samples in zip(self.hmc_helper_grw_nf[year_index[i]].current_state, self.samples_grw_nf[year_index[i]]):
                var.assign(var_samples[index[2]])
            # fec
            for var, var_samples in zip(self.hmc_helper_fec[year_index[i]].current_state, self.samples_fec[year_index[i]]):
                var.assign(var_samples[index[3]])
            # flow
            for var, var_samples in zip(self.hmc_helper_flow[year_index[i]].current_state, self.samples_flow[year_index[i]]):
                var.assign(var_samples[index[4]])

            models_now2 = {
                "m_sur": self.m_sur_new[year_index[i]],
                "m_grw_f": self.m_grw_f_new[year_index[i]],
                "m_grw_nf": self.m_grw_nf_new[year_index[i]],
                "m_fec": self.m_fec_new[year_index[i]],
                "m_flow_poi": self.m_flow_poi_new[year_index[i]],
                "alpha": self.alpha_array[year_index[i]],
                "beta": self.beta_array[year_index[i]],
                "recruit_p": self.recruit_p_array[year_index[i]]
            }

            data_simu = IBM_1step(zt=current_z[0], age=current_z[1], models=models_now2)
            current_z = popu_structure_noid(data_simu)

            n_new1 = np.sum(data_simu["surv"] == 1)
            n_new2 = np.sum(np.isnan(data_simu["size"])) 
            n_old = np.sum(np.logical_not(np.isnan(data_simu["size"])))

            # growth rate
            l_ABC.append((n_new1+n_new2)/n_old)
            # population size
            n_ABC.append(n_new1+n_new2)


        return (l, n, l_ABC, n_ABC, year_index)  



@ray.remote
class para_ipm_sep():
    def __init__(self, ipm_sep):
        # m is an object belonging to ipm_sep
        self.ipm_sep = ipm_sep
    
    def random_lambda(self):
        return self.ipm_sep.random_lambda()

    def random_lambda_full(self):
        return self.ipm_sep.random_lambda_full()



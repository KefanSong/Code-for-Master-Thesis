import os
import pickle
import torch
from utils.misc import println

class Logger:

    def __init__(self, hyperparams, group, comment=''):

        self.log_data = {'time': [],
                         'MinR': [],
                         'MaxR': [],
                         'AvgR': [],
                         'AvgR2': [],
                         'MinC': [],
                         'MaxC': [],
                         'AvgC': [],
                         'nu0': [],
                         'nu1': [],
                         'running_stat': None}

        self.models = {'iter': None,
                       'policy_params': None,
                       'value_params': [None, None],
                       'cvalue_params': None,
                       'pi_optimizer': None,
                       'vf_optimizer': [None,None],
                       'cvf_optimizer': None,
                       'pi_loss': None,
                       'vf_loss': [None,None],
                       'cvf_loss': None}

        self.hyperparams = hyperparams
        self.iter = 0
        self.group = group
        self.comment = comment

        



    def update(self, key, value):
        if type(self.log_data[key]) is list:
            self.log_data[key].append(value)
        else:
            self.log_data[key] = value

    def save_model(self, component, params):
        self.models[component] = params


    def load_model(self):
        constraint = self.hyperparams['constraint']
        seed = self.hyperparams['seed']
        
        

        # envname = env_id.partition(':')[-1] if ':' in env_id else env_id
        envnames = ['HalfCheetah-v4', 'BigFootHalfCheetah']
        envname = envnames[self.group]
        # create a different log for different group 
        # 'group', str(self.group),
        directory = '_'.join(['focops', 'results'])
        filename1 = '_'.join(['focops',  constraint, str(self.group),envname,  'log_data_seed', str(seed)]) + self.comment+ '.pkl'
        filename2 = '_'.join(['focops',  constraint, str(self.group),envname,  'hyperparams_seed', str(seed)]) + '.pkl'
        filename3 = '_'.join(['focops',  constraint, str(self.group),envname,  'models_seed', str(seed)]) + '.pth'

        if not os.path.exists(directory):
            os.mkdir(directory)

        # pickle.dump(self.log_data, open(os.path.join(directory, filename1), 'wb'))
        # pickle.dump(self.hyperparams, open(os.path.join(directory, filename2), 'wb'))
        self.models = torch.load(os.path.join(directory, filename3))

        # Also load log data
        self.log_data = pickle.load(open(os.path.join(directory, filename1), 'rb'))
        
        


    def dump(self):
        batch_size = self.hyperparams['batch_size']
        # Print results
        println('Results for Iteration:', self.iter + 1)
        println('Number of Samples:', (self.iter + 1) * batch_size)
        println('Time: {:.2f}'.format(self.log_data['time'][-1]))
        println('MinR: {:.2f}| MaxR: {:.2f}| AvgR: {:.2f}| AvgR2: {:.2f}'.format(self.log_data['MinR'][-1],
                                                                  self.log_data['MaxR'][-1],
                                                                  self.log_data['AvgR'][-1],
                                                                  self.log_data['AvgR2'][-1]))
        println('MinC: {:.2f}| MaxC: {:.2f}| AvgC: {:.2f}'.format(self.log_data['MinC'][-1],
                                                                  self.log_data['MaxC'][-1],
                                                                  self.log_data['AvgC'][-1]))
        println('Nu0: {:.3f}'.format(self.log_data['nu0'][-1]))
        println('Nu1: {:.3f}'.format(self.log_data['nu1'][-1]))
        println('--------------------------------------------------------------------')


        # Save Logger
        # env_id = self.hyperparams['env_id']
        constraint = self.hyperparams['constraint']
        seed = self.hyperparams['seed']
        
        

        # envname = env_id.partition(':')[-1] if ':' in env_id else env_id
        envnames = ['HalfCheetah-v4', 'BigFootHalfCheetah']
        envname = envnames[self.group]
        # create a different log for different group 
        # 'group', str(self.group),
        directory = '_'.join(['foc3333ops', 'results'])
        filename1 = '_'.join(['focops',  constraint, str(self.group),envname,  'log_data_seed', str(seed)]) + self.comment+ '.pkl'
        filename2 = '_'.join(['focops',  constraint, str(self.group),envname,  'hyperparams_seed', str(seed)])  + '.pkl'
        filename3 = '_'.join(['focops',  constraint, str(self.group),envname,  'models_seed', str(seed)]) + '.pth'

        if not os.path.exists(directory):
            os.mkdir(directory)

        pickle.dump(self.log_data, open(os.path.join(directory, filename1), 'wb'))
        pickle.dump(self.hyperparams, open(os.path.join(directory, filename2), 'wb'))
        torch.save(self.models, os.path.join(directory, filename3))

        # Advance iteration by 1
        self.iter += 1

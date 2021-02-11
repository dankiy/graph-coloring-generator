import os
import networkx as nx

from abc import ABCMeta, abstractmethod
from joblib import Parallel, delayed
from pydoc import locate
from scipy.stats import truncnorm, uniform
from networkx.generators import interval_graph, erdos_renyi_graph, \
    barabasi_albert_graph, watts_strogatz_graph


class BaseGenerator(metaclass=ABCMeta):
    def __init__(self, num_samples, subfolder, params, seed):
        self.num_samples = num_samples
        self.subfolder = subfolder 
        self.params = params
        self.seed = seed

    def _get_instance_params(self):
        instance_params = {}
        for key, val in self.params.items():
            if val['distribution'] == 'normal':
                p = truncnorm(*val['range']).rvs()
            elif val['distribution'] == 'uniform':
                p = uniform(*val['range']).rvs()
            instance_params[key] = locate(val['type'])(p)
        return instance_params            

    @abstractmethod
    def _generate(self):
        raise NotImplementedError

    def _save_instance(self, save_path):
        G = self._generate()
        nx.write_gpickle(G, save_path)

    def save(self, output_path, num_workers):
        if not os.path.exists(f'{output_path}/{self.subfolder}'):
            os.makedirs(f'{output_path}/{self.subfolder}')
        Parallel(n_jobs=num_workers, verbose=0)(
            delayed(self._save_instance)(f'{output_path}/{self.subfolder}/{str(idx)}.gpickle')
                for idx in range(self.num_samples))   

class IntervalGenerator(BaseGenerator):
    def _generate(self):
        instance_params = self._get_instance_params()
        intervals = [sorted(uniform.rvs(size=2)) for _ in range(instance_params['n'])]
        G = interval_graph(intervals)
        return G

class ErdosRenyiGenerator(BaseGenerator):
    def _generate(self):
        instance_params = self._get_instance_params()
        G = erdos_renyi_graph(**instance_params)
        return G

class BarabasiAlbertGenerator(BaseGenerator):
    def _generate(self):
        instance_params = self._get_instance_params()
        G = barabasi_albert_graph(**instance_params)
        return G

class WattsStrogatzGenerator(BaseGenerator):
    def _generate(self):
        instance_params = self._get_instance_params()
        G = watts_strogatz_graph(**instance_params)
        return G
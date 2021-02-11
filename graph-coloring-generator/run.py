import numpy as np

from argparse import ArgumentParser
from tqdm import tqdm
from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from core import IntervalGenerator, ErdosRenyiGenerator, BarabasiAlbertGenerator, WattsStrogatzGenerator


gen_dict = {
    'interval' : IntervalGenerator,    
    'erdos_renyi' : ErdosRenyiGenerator,
    'barabasi_albert' : BarabasiAlbertGenerator,
    'watts_strogatz' : WattsStrogatzGenerator
}


def main(args):
    stream = open(args.config)
    config = load(stream, Loader=Loader)

    np.random.seed(config['seed'])

    for gen_name, gen_args in config['generators'].items():
        print(f'\n{gen_name}\n{gen_args}')
        gen = gen_dict[gen_name](**gen_args, seed=config['seed'])
        gen.save(config['output_path'], config['num_workers'])
    print('\nDone \n')


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(args)

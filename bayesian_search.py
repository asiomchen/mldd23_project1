from bayes_opt import BayesianOptimization
from src.disc.discriminator import Discriminator
import torch
import pandas as pd
import argparse
import numpy as np
import multiprocessing as mp
import queue
import time
import random


class Scorer:
    """
    Scorer class for Bayesian optimization
    """
    def __init__(self, path, latent_size, boundary, penalize=False, device='cpu'):
        """
        Args:
            path: path to the discriminator model
            latent_size: size of the latent space
            boundary: value of the upper and lower bound for the latent space search
            penalize: if True, penalize for values outside of bounds
        """
        self.model = Discriminator(latent_size=latent_size, use_sigmoid=True).to(device)
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.bounds = boundary
        self.penalize = penalize

    def __call__(self, **args) -> float:
        input_tensor = torch.tensor(list({**args}.values()))
        input_tensor = input_tensor.to(torch.float32)
        pred = self.model(input_tensor)
        output = pred.cpu().detach().numpy()[0]
        if self.penalize:
            output = output * (1 - self.penalty(input_tensor, self.bounds))
        return output

    def penalty(self, tensor, boundary) -> float:
        """
        Penalize for values outside of bounds
        Args:
            tensor (torch.Tensor): latent tensor
            boundary (float): value of the upper and lower bound for the latent space search
        Returns:
            float: penalty
        """
        dist = torch.pow(tensor, 2).sum()
        if dist < boundary/2:
            penalty = 0
        elif dist > boundary:
            penalty = 1
        else:
            penalty = boundary - dist
        return penalty


def search(parser, return_list):
    """
    Perform Bayesian optimization on the latent space in respect to the discriminator output
    Args:
        parser: argparse parser object
        return_list: list to append results to (multiprocessing)
    Returns:
        None
    """
    # parse arguments
    args = parser.parse_args()

    # initialize scorer
    latent_size = 32
    scorer = Scorer(args.model_path, latent_size, args.bounds)

    # define bounds
    pbounds = {str(p): (-scorer.bounds, scorer.bounds) for p in range(latent_size)}

    # initialize optimizer
    random_state = random.randint(0, 1000000)
    optimizer = BayesianOptimization(
        f=scorer,
        pbounds=pbounds,
        random_state=random_state,
        verbose=args.verbose
    )

    vector_list = []
    score_list = []

    # run optimization:
    optimizer.maximize(
        init_points=args.init_points,
        n_iter=args.n_iter,
    )
    vector = np.array(list(optimizer.max['params'].values()))
    score_list.append(float(optimizer.max['target']))
    vector_list.append(vector)

    # append results to return list
    samples = pd.DataFrame(np.array(vector_list))
    samples.columns = [str(n) for n in range(latent_size)]
    samples['score'] = score_list
    samples['score'] = samples['score'].astype(float)
    return_list.append(samples)
    return None


if __name__ == '__main__':
    random.seed(42)
    """
    Multiprocessing support and queue handling
    """
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument('-n', '--n_samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('-p', '--init_points', type=int, default=1,
                        help='Number of initial points to sample')
    parser.add_argument('-i', '--n_iter', type=int, default=8,
                        help='Number of iterations to perform')
    parser.add_argument('-b', '--bounds', type=float, default=2.0,
                        help='Bounds for the latent space search')
    parser.add_argument('-v', '--verbose', type=bool, default=False,
                        help='Verbosity')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help='Device to use for search')

    samples = pd.DataFrame()  # placeholder
    args = parser.parse_args()
    device = torch.device(args.device)
    n_samples = args.n_samples

    if args.device == 'cpu':
        manager = mp.Manager()
        return_list = manager.list()
        cpus = mp.cpu_count()
        print("Number of cpus: ", cpus)

        queue = queue.Queue()

        for i in range(n_samples):
            proc = mp.Process(target=search, args=[parser,  return_list])
            queue.put(proc)

        # handle the queue
        processes = []
        while True:
            if queue.empty():
                print("(mp) Queue handled successfully")
                break
            if len(mp.active_children()) < cpus:
                proc = queue.get()
                proc.start()
                if queue.qsize() % 5 == 0:
                    print('(mp) Processes in queue: ', queue.qsize())
                processes.append(proc)
            time.sleep(1)

        # complete the processes
        for proc in processes:
            proc.join()

        samples = pd.concat(return_list)
        end_time = time.time()
        time_elapsed = end_time - start_time
        print("Time elapsed: ", round(time_elapsed, 2), "s")

    elif args.device == 'cuda':
        raise NotImplementedError

    # save the results
    print(samples.head())
    samples.to_parquet('results/samples.parquet', index=False)

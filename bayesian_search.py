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

class Scorer():
    def __init__(self, latent_size, bounds, penalize=False):
        self.model = Discriminator(latent_size=latent_size, use_sigmoid=True).to(device)
        self.model.load_state_dict(
            torch.load('models/discr_d2_tatra_epoch_80/epoch_150.pt', map_location=device)
        )
        self.bounds = bounds
        self.penalize = penalize

    def __call__(self, **args) -> float:
        input_tensor = torch.tensor(list({**args}.values()))
        input_tensor = input_tensor.to(torch.float32)
        pred = self.model(input_tensor)
        output = pred.cpu().detach().numpy()[0]
        if self.penalize:
            output = output * self.penalty(input_tensor)
        return output

    def penalty(self, tensor):
        """
        Penalize for values outside of bounds
        Args:
            tensor: latent tensor
        Returns:
            float: penalty
        """
        total_penalty = 1
        for i in range(len(tensor)):
            x = abs(tensor[i])
            if x > self.bounds/2:
                total_penalty *= (x - self.bounds/2) / x
        return total_penalty


def search(n_samples, init_points, n_iter, return_list, bounds, verbose):

    latent_size = 32
    scorer = Scorer(latent_size, bounds)
    pbounds = {str(p): (-scorer.bounds, scorer.bounds) for p in range(latent_size)}
    random_state = random.randint(0, 1000000)

    optimizer = BayesianOptimization(
        f=scorer,
        pbounds=pbounds,
        random_state=random_state,
        verbose=verbose
    )
    vector_list = []
    score_list = []
    for _ in range(n_samples):
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter
        )
        vector = np.array(list(optimizer.max['params'].values()))
        score_list.append(float(optimizer.max['target']))
        print("Score: ", optimizer.max['target'])
        vector_list.append(vector)

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
    parser.add_argument('-n', '--n_samples', type=int, default=10)
    parser.add_argument('-p', '--init_points', type=int, default=1)
    parser.add_argument('-i', '--n_iter', type=int, default=8)
    parser.add_argument('-b', '--bounds', type=float, default=2.0)
    n_samples = parser.parse_args().n_samples
    init_points = parser.parse_args().init_points
    n_iter = parser.parse_args().n_iter
    bounds = parser.parse_args().bounds
    samples = pd.DataFrame()
    verbose = False
    device = torch.device('cpu')

    if device == torch.device('cpu'):
        manager = mp.Manager()
        return_list = manager.list()
        score_list = manager.list()

        cpus = mp.cpu_count()
        print("Number of cpus: ", cpus)

        queue = queue.Queue()

        # prepare a process for each file and add to queue

        if n_samples < cpus:
            for i in range(n_samples):
                proc = mp.Process(target=search, args=(n_samples, init_points, n_iter,
                                                       return_list, bounds, verbose))
                queue.put(proc)
        else:
            chunk_size = n_samples // cpus
            remainder = n_samples % cpus
            for i in range(cpus):
                if i > cpus - remainder:
                    proc = mp.Process(target=search, args=(chunk_size + 1, init_points, n_iter,
                                                           return_list, bounds, verbose))
                else:
                    proc = mp.Process(target=search, args=(chunk_size, init_points, n_iter,
                                                           return_list, bounds, verbose))
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
                processes.append(proc)
            time.sleep(1)

        # complete the processes
        for proc in processes:
            proc.join()

        samples = pd.concat(return_list)
        print(samples)
        end_time = time.time()
        time_elapsed = end_time - start_time
        print("Time elapsed: ", round(time_elapsed, 2), "s")
    # save the results
    print(samples.head())
    samples.to_parquet('results/samples.parquet', index=False)
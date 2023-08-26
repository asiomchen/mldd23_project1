from bayes_opt import BayesianOptimization
from src.disc.discriminator import Discriminator
import torch
import pandas as pd
import argparse
import numpy as np
import multiprocessing as mp
import queue
import time

class Scorer():
    def __init__(self, latent_size, device):
        self.model = Discriminator(latent_size=latent_size, use_sigmoid=False).to(device)
        self.model.load_state_dict(
            torch.load('models/discr_d2_mandarynka_epoch_100/epoch_150.pt', map_location=device)
        )

    def __call__(self, **args) -> float:
        input_tensor = torch.tensor(list({**args}.values()))
        input_tensor = input_tensor.to(torch.float32)
        pred = self.model(input_tensor)
        output = pred.cpu().detach().numpy()[0]
        return output


def search(n_samples, init_points, n_iter, return_list, random_state=42):

    device = torch.device('cpu')
    latent_size = 64
    scorer = Scorer(latent_size, device)

    pbounds = {str(p): (-10, 10) for p in range(latent_size)}

    optimizer = BayesianOptimization(
        f=scorer,
        pbounds=pbounds,
        random_state=random_state,
        verbose=0
    )
    vector_list = []
    for _ in range(n_samples):
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter
        )
        vector = np.array(list(optimizer.max['params'].values()))
        vector_list.append(vector)
    samples = pd.DataFrame(np.array(vector_list))
    samples.columns = [str(n) for n in range(latent_size)]
    return_list.append(samples)
    return None

if __name__ == '__main__':
    """
    Multiprocessing support and queue handling
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_samples', type=int, default=10)
    parser.add_argument('-p', '--init_points', type=int, default=1)
    parser.add_argument('-i', '--n_iter', type=int, default=8)
    n_samples = parser.parse_args().n_samples
    init_points = parser.parse_args().init_points
    n_iter = parser.parse_args().n_iter

    manager = mp.Manager()
    return_list = manager.list()

    cpus = mp.cpu_count()
    print("Number of cpus: ", cpus)

    queue = queue.Queue()
    verbose = True

    # prepare a process for each file and add to queue

    if n_samples < cpus:
        for i in range(n_samples):
            proc = mp.Process(target=search, args=(n_samples, init_points, n_iter, return_list))
            queue.put(proc)
    else:
        chunk_size = n_samples // cpus
        remainder = n_samples % cpus
        for i in range(cpus):
            if i > cpus - remainder:
                proc = mp.Process(target=search, args=(chunk_size + 1, init_points, n_iter, return_list))
            else:
                proc = mp.Process(target=search, args=(chunk_size, init_points, n_iter, return_list))
            queue.put(proc)

    # handle the queue
    processes = []
    while True:
        if queue.empty():
            print("(mp) Queue handled successfully")
            break
        if len(mp.active_children()) < cpus:
            proc = queue.get()
            print("(mp) Starting:", proc.name)
            proc.start()
            processes.append(proc)
        time.sleep(1)

    # complete the processes
    for proc in processes:
        proc.join()

    # save the results
    samples = pd.concat(return_list)
    samples.to_parquet('data/samples.parquet', index=False)
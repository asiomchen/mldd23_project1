from bayes_opt import BayesianOptimization
from src.clf.scorer import MLPScorer, SKLearnScorer
import pandas as pd
import argparse
import numpy as np
import multiprocessing as mp
import queue
import time
import random
import warnings

# suppress scikit-learn warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn


def search(args, return_list):
    """
    Perform Bayesian optimization on the latent space in respect to the discriminator output
    Args:
        args: dictionary of arguments (argparse)
            contains:
                model_path: path to the model
                model_type: type of the model (mlp or sklearn)
                n_samples: number of samples to generate
                init_points: number of initial points to sample
                n_iter: number of iterations to perform
                bounds: bounds for the latent space search
                verbose: verbosity
                latent_size: size of the latent space
        return_list: list to append results to (multiprocessing)
    Returns:
        None
    """

    # initialize scorer
    latent_size = args.latent_size
    if args.model_type == 'mlp':
        scorer = MLPScorer(args.model_path, latent_size, penalize=False)
    elif args.model_type == 'sklearn':
        scorer = SKLearnScorer(args.model_path, penalize=False)
    else:
        raise ValueError("Model type must be either 'mlp' or 'sklearn")

    # define bounds
    pbounds = {str(p): (-args.bounds, args.bounds) for p in range(latent_size)}

    random_state = random.randint(0, 1000000)

    # initialize optimizer
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
    parser.add_argument('-t', '--model_type', type=str, default='mlp',
                        help='Model type: mlp or sklearn')
    parser.add_argument('-n', '--n_samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('-p', '--init_points', type=int, default=1,
                        help='Number of initial points to sample')
    parser.add_argument('-i', '--n_iter', type=int, default=8,
                        help='Number of iterations to perform')
    parser.add_argument('-b', '--bounds', type=float, default=1.0,
                        help='Bounds for the latent space search')
    parser.add_argument('-v', '--verbose', type=bool, default=False,
                        help='Verbosity')
    parser.add_argument('-l', '--latent_size', type=int, default=32)

    samples = pd.DataFrame()  # placeholder
    args = parser.parse_args()
    n_samples = args.n_samples

    manager = mp.Manager()
    return_list = manager.list()
    cpus = mp.cpu_count()
    print("Number of cpus: ", cpus)

    queue = queue.Queue()

    for i in range(n_samples):
        proc = mp.Process(target=search, args=[args, return_list])
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
    time_elapsed = (end_time - start_time) / 60  # in minutes
    if time_elapsed < 60:
        print("Time elapsed: ", round(time_elapsed, 2), "min")
    else:
        print("Time elapsed: ", int(time_elapsed // 60), "h", round(time_elapsed % 60, 2), "min")

    # save the results
    model_name = args.model_path.split('/')[-1]
    samples.to_parquet(f'results/{model_name}.parquet', index=False)

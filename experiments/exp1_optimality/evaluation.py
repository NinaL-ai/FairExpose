from helper import generate_dataset, partial_normalization
from metrics import get_metrics
from ranking import RankingProblem
from ranking import FairExpose
from ranking import fair_norm_rank

import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
import time
from concurrent.futures import wait, FIRST_EXCEPTION
import sys

def get_topk(method, problem):
    ranker = FairExpose(problem)
    # proportional and exposure fairness
    if method == "FairExpose-Pro":
        start_time = time.time()
        ranker.fair_expose_pro()
        top_k_df = ranker.topk
        preprocessing_time = time.time() - start_time

    # ordered and exposure fairness
    elif method == "FairExpose-Ord":
        start_time = time.time()
        ranker.fair_expose_ord()
        top_k_df = ranker.topk
        preprocessing_time = time.time() - start_time
    elif method == "FairExpose-Ord Greedy":
        start_time = time.time()
        ranker.fair_expose_ord_greedy()
        top_k_df = ranker.topk
        preprocessing_time = time.time() - start_time

    # Baseline and methods from other authors
    elif method == "FairNormRank":
        start_time = time.time()
        data = partial_normalization(problem.candidates, problem.relevance_col, problem.group_col, alpha=1.0)
        top_k_df = fair_norm_rank(data, problem.k, problem.proportional_target, problem.group_col,
                                  problem.relevance_col + "_norm")
        preprocessing_time = time.time() - start_time
    elif method == "colorblind":
        start_time = time.time()
        top_k_df = (problem.candidates.sort_values(by=problem.relevance_col, ascending=False)
                    .head(problem.k)
                    .reset_index(drop=True))
        preprocessing_time = time.time() - start_time
    elif method == "random":
        start_time = time.time()
        top_k_df = problem.candidates.sample(n=problem.k, replace=False, random_state=None)
        preprocessing_time = time.time() - start_time
    else:
        print("Unknown method")

    return top_k_df, preprocessing_time


def save_results(top_k_df, metrics, method, dataset, process_time, k, trial):
    rank_path = "results/ranking/"
    metric_path = "results/metrics/"
    file_prefix = method + "_" + dataset + "_k" + str(k) + "_t" + str(trial) + "_"

    if not os.path.exists(rank_path):
        os.makedirs(rank_path)

    if not os.path.exists(metric_path):
        os.makedirs(metric_path)

    top_k_df.to_csv(rank_path + file_prefix + "top_k_df.csv")

    metrics["runtime"] = process_time
    metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
    metrics_df.to_csv(metric_path + file_prefix + "metrics.csv")


def run_single_experiment(method, problem, trial, dataset_string):
    print("--------------", method, dataset_string, trial)

    # define target proportions
    top_k_df, process_time = get_topk(method, problem)
    metrics = get_metrics(problem, top_k_df)
    save_results(top_k_df, metrics, method, dataset_string, process_time, problem.k, trial)
    print("--------------finished", method, dataset_string, trial)


def run_single_experiment_wrapper(method, k, trial, dataset_name):
    try:
        if dataset_name == "binary":
            data, rel_score, prot_attr = generate_dataset(500, {"Gender": 2}, unfairness=0.05)
        elif dataset_name == "non-binary":
            data, rel_score, prot_attr = generate_dataset(500, {"Race": 4}, unfairness=0.05)
        elif dataset_name == "multi binary":
            data, rel_score, prot_attr = generate_dataset(500, {"Gender": 2, "Age": 2},
                                                                     unfairness=0.05)
        elif dataset_name == "multi non-binary":
            data, rel_score, prot_attr = generate_dataset(500, {"Race": 3, "Age": 3},
                                                          unfairness=0.05)

        z_unique = data[prot_attr].unique()
        target_proportions = {key: 1 / len(z_unique) for key in z_unique}
        problem = RankingProblem(data, prot_attr, rel_score, k=k, proportional_target=target_proportions)

        return run_single_experiment(method, problem, trial, dataset_name)
    except Exception as e:
        print(f"\n Experiment failed for method='{method}', dataset='{dataset_name}', trial={trial}")
        raise

def start_experiment_parallel(methods, datasets, trials=10, max_workers=None):
    tasks = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        try:
            for dataset_string in datasets:
                k = datasets[dataset_string]

                for method in methods:
                    for trial in range(trials):
                        tasks.append(
                            executor.submit(
                                run_single_experiment_wrapper,
                                method,
                                k,
                                trial,
                                dataset_string
                            )
                        )

            # Wait for tasks and fail fast
            done, not_done = wait(tasks, return_when=FIRST_EXCEPTION)

            for future in done:
                if future.exception():
                    raise future.exception()

        except Exception as e:
            print(f"\n Aborting: Experiment pool failed with error: {e}", file=sys.stderr)
            executor.shutdown(wait=False, cancel_futures=True)
            raise

if __name__ == "__main__":
    # start experiment
    methods = ["colorblind", "FairExpose-Pro", "FairExpose-Ord","FairExpose-Ord Greedy"]
    datasets = {
        "binary": 100,
        "non-binary": 100,
        "multi binary": 100,
        "multi non-binary": 200
        }

    start_experiment_parallel(methods, datasets, trials=1)

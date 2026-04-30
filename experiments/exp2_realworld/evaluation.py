from helper import *
from metrics import get_metrics
from ranking import FairExpose
from ranking import RankingProblem
from ranking import fair_norm_rank

import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
import time
from concurrent.futures import wait, FIRST_EXCEPTION
import sys

def get_topk(method, problem):
    ranker = FairExpose(problem)
    if method == "FairExpose-Pro":  # proportional and exposure fairness
        start_time = time.time()
        ranker.fair_expose_pro()
        top_k_df = ranker.topk
        preprocessing_time = time.time() - start_time
    elif method == "FairExpose-Ord":  # ordered and exposure fairness
        start_time = time.time()
        ranker.fair_expose_ord()
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


def save_results(top_k_df, metrics, method, dataset, protected_name, process_time, k, trial):
    path = "metrics/" + method + "/"
    file_prefix = dataset + "_k" + str(k) + "_t" + str(trial) + "_"

    if not os.path.exists(path):
        os.makedirs(path)

    if trial == 0:
        top_k_df.to_csv(path + file_prefix + "top_k_df.csv")
    counts_df = top_k_df[protected_name].value_counts(dropna=False, normalize=True)
    metrics["group_counts"] = counts_df.to_list()
    metrics["runtime"] = process_time
    metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
    metrics_df.to_csv(path + file_prefix + "metrics.csv")

def run_single_experiment(method, k, trial, dataset_string, data, relevance_score, protected_attr):
    print("--------------", method, dataset_string, trial)

    # define target proportions
    groups = data[protected_attr].unique()
    targets = {key: 0.0 for key in groups}
    if dataset_string == "Compas_Gender":
        targets["Female"] = 0.5
    elif dataset_string == "German_Gender":
        targets["female"] = 0.5
    elif dataset_string == "LSAT_Gender":
        targets["female"] = 0.5
    elif dataset_string == "Compas_Race":
        targets["African-American"] = 1/7
        targets['Asian'] = 1/7
        targets['Native American'] = 1/7
        targets['Oriental'] = 1/7
        targets['Arabic'] = 1/7
    elif dataset_string == "German_Age":
        targets["young"] = 0.34
        targets["old"] = 0.34
    elif dataset_string == "German_Age,Gender":
        targets["young_female"] = 0.167
        targets["young_male"] = 0.167
        targets["old_female"] = 0.167
        targets["old_male"] = 0.167
    elif dataset_string == "LSAT_Race":
        targets["black"] = 0.25
        targets["hisp"] = 0.25
        targets["asian"] = 0.25
    elif dataset_string == "LSAT_Race,Gender":
        targets["black_female"] = 0.125
        targets["black_male"] = 0.125
        targets["hisp_female"] = 0.125
        targets["hisp_male"] = 0.125
        targets['asian_male'] = 0.125
        targets['asian_female'] = 0.125

    problem = RankingProblem(data, protected_attr, relevance_score, k, targets)

    top_k_df, process_time = get_topk(method, problem)
    metrics = get_metrics(problem, top_k_df)
    save_results(top_k_df, metrics, method, dataset_string, protected_attr, process_time, k, trial)
    print("--------------finished", method, dataset_string, trial)


def run_single_experiment_wrapper(method, k, trial, dataset_string):
    try:
        dataset_name = dataset_string.split("_")[0]
        protected_name = dataset_string.split("_")[1]
        data, relevance_score, protected_attr = load_data(dataset_name, protected=protected_name)
        remove = ["other", "Other", "other_female", "other_male"]
        data = data[~data[protected_attr].isin(remove)]

        return run_single_experiment(
            method, k, trial, dataset_string, data, relevance_score, protected_attr
        )
    except Exception as e:
        print(f"\n Experiment failed for method='{method}', dataset='{dataset_string}', trial={trial}")
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
    methods = [
        "colorblind",
        "random",
        "FairNormRank",
        "FairExpose-Pro",
        "FairExpose-Ord"
    ]

    # dataset string to k value mapping
    datasets = {
        "Compas_Gender": 500,
        "Compas_Race": 90,
        "German_Gender": 400,
        "German_Age": 200,
        "German_Age,Gender": 200,
        "LSAT_Gender": 2000,
        "LSAT_Race": 500,
        "LSAT_Race,Gender": 300,
    }
    start_experiment_parallel(methods, datasets, trials=10)

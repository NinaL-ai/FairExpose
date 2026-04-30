from helper.dataset import *
import pandas as pd
import uuid
import os

def prepare_dataset(dataset):
    print("processing ", dataset)
    dataset_name, protected_name = dataset.split("_")
    data, relevance_score, protected_attr = load_data(dataset_name=dataset_name, protected=protected_name)
    remove = ["other", "Other", "other_female", "other_male"]
    data = data[~data[protected_attr].isin(remove)]

    #score,group,uuid
    data['uuid'] = [str(uuid.uuid4()) for _ in range(len(data))]
    data['score'] = data[relevance_score]

    # group = 0 for unprotected and > 0 for protected
    if dataset == "Compas_Gender":
        pro = ["Female"]
    elif dataset == "German_Gender":
        pro = ["female"]
    elif dataset == "LSAT_Gender":
        pro = ["female"]

    elif dataset == "Compas_Race":
        pro = ["African-American", 'Asian', 'Native American', 'Oriental', 'Arabic']
    elif dataset == "German_Age":
        pro = ["young", "old"]  ##
    elif dataset == "German_Age,Gender":
        pro = ["young_female", "young_male", "old_female", "old_male"]   ##
    elif dataset == "LSAT_Race":
        pro = ["black", "hisp", "asian"]   ##
    elif dataset == "LSAT_Race,Gender":
        pro = ["black_female", "black_male", "hisp_female", "hisp_male", "asian_female", "asian_male"]  ##
    values = data[protected_attr].astype(str)
    group_mapping = {val: i + 1 for i, val in enumerate(pro)}  # protected values start from 1
    data['group'] = values.map(group_mapping).fillna(0).astype(int)  #.fillna(0) for unprotected

    print(dataset_name, protected_name, len(data["group"].unique()))

    data = data.sort_values(by=relevance_score, ascending=False)

    data.to_csv("datasets/{}_{}_orig.csv".format(dataset_name,protected_name), index=False, sep=",")
    data = data[["score", "group", "uuid"]]
    data.to_csv("datasets/{}_{}_java.csv".format(dataset_name,protected_name), index=False, sep=",")

if __name__ == "__main__":
    # os.chdir("../../")
    # print(os.getcwd())
    datasets = [
        # "Compas_Gender", "Compas_Race",
        # "German_Gender", "German_Age", "German_Age,Gender",
        "LSAT_Gender", "LSAT_Race", "LSAT_Race,Gender"
    ]
    for dataset in datasets:
        prepare_dataset(dataset)
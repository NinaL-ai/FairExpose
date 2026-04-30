from metrics import *
from helper.dataset import load_data
import os
import pandas as pd


def top_k_to_orig_ids():
    # Directories
    results_dir = "results_raw"
    datasets_dir = "datasets"

    # Normalize filenames for case-insensitive lookup
    orig_files = {
        os.path.splitext(f.lower().replace("_orig", ""))[0]: os.path.join(datasets_dir, f)
        for f in os.listdir(datasets_dir)
        if f.endswith("_orig.csv")
    }

    # Process results_raw
    for file in os.listdir(results_dir):
        if not file.endswith(".csv"):
            continue
        if file.startswith("runtime"):
            continue
        if file.endswith("remaining.csv") or file.endswith("unfair.csv"):
            continue

        base_name = os.path.splitext(file)[0]

        # Try to split only on the first underscore
        try:
            ds_name_part = base_name.split("_")[0]
            group_name_part = base_name.split("_")[1]
        except ValueError:
            print(f"Skipping malformed filename: {file}")
            continue

        key = f"{ds_name_part.lower()}_{group_name_part.lower()}"
        orig_file_path = orig_files.get(key)

        if not orig_file_path or not os.path.exists(orig_file_path):
            print(f"No matching original file for: {file}")
            continue

        # Load DataFrames
        df_results = pd.read_csv(os.path.join(results_dir, file))
        df_orig = pd.read_csv(orig_file_path)

        # Merge on uuid
        df_merged = df_results.merge(df_orig, on='uuid', how='left')

        # Save as new file with _matched.csv suffix
        output_filename = f"{base_name}_matched.csv"
        output_path = os.path.join("results_matched", output_filename)
        df_merged.to_csv(output_path, index=False)
        print(f"Saved matched file: {output_path}")


def calc_metrics(dataset_name):
    path = "results_matched"
    for file in os.listdir(path):
        ds_name = file.split("_")[0]
        protected_name = file.split("_")[1]
        dataset_string = ds_name + "_" + protected_name
        runtime_file = "runtime_results_" + ds_name + "_" + protected_name + ".csv"
        ds_name = ds_name.capitalize()
        #protected_name = protected_name.capitalize()
        if ds_name == "Lsat":
            ds_name = "LSAT"

        if ds_name == dataset_name:
            print(".....Processing", file)
            # define target_proportions
            #target = file.split("_")[3].split("=")[1]
            #target = ast.literal_eval(target)
            data, relevance_score, protected_attr = load_data(dataset_name=ds_name, protected=protected_name.capitalize())
            remove = ["other", "Other", "other_female", "other_male"]
            data = data[~data[protected_attr].isin(remove)]
            # groups = data[protected_attr].unique()
            # target_proportions = {}
            # for group in groups:
            #     target_proportions[group] = 1 / len(groups)

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


            top_k_df = pd.read_csv(os.path.join(path, file))

            # runtime
            trial = int(file.split("_")[-2].split("t")[1])
            runtime_df = pd.read_csv(os.path.join("results_raw", runtime_file))
            runtime = runtime_df[runtime_df["Trial"] == trial]["Time(s)"].iloc[0]
            k = len(top_k_df)

            metrics = get_metrics(top_k_df, data,
                                  target_proportions=targets,
                                  protected_attr="z",
                                  relevance_score=relevance_score)

            counts_df = top_k_df["z"].value_counts(dropna=False, normalize=True)
            metrics["group_counts"] = counts_df.to_list()

            metrics["runtime"] = runtime
            metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
            fname = ds_name + "_" + protected_name + "_k" + str(k) + "_t" + str(trial-1) + "_metrics.csv"
            metrics_df.to_csv("../metrics/zehlike/" + fname)


if __name__ == "__main__":
    top_k_to_orig_ids()
    calc_metrics(dataset_name="Compas")
    calc_metrics(dataset_name="German")
    calc_metrics(dataset_name="LSAT")

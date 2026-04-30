import os
import re
import pandas as pd
from collections import defaultdict

def aggregate_metrics(base_dir="metrics"):
    pattern = re.compile(r"(.+?)_(.+?)_k(\d+)_t\d+_metrics\.csv")

    for method in os.listdir(base_dir):
        method_path = os.path.join(base_dir, method)
        if not os.path.isdir(method_path):
            continue

        # key = (dataset, group, k), value = list of trial DataFrames
        grouped_metrics = defaultdict(list)

        for file in os.listdir(method_path):
            match = pattern.match(file)
            if not match:
                continue

            dataset, group, k = match.groups()
            key = (dataset, group, k)
            file_path = os.path.join(method_path, file)

            try:
                df = pd.read_csv(file_path)
                df = df[['Metric', 'Value']]
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
                df = df.dropna(subset=['Value'])
                df = df.set_index('Metric')
                grouped_metrics[key].append(df)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        for (dataset, group, k), dfs in grouped_metrics.items():
            if not dfs:
                continue

            combined = pd.concat(dfs, axis=1, join='inner')  # align by metric
            combined.columns = [f"trial_{i}" for i in range(len(dfs))]

            result = pd.DataFrame({
                'mean': combined.mean(axis=1),
                'std': combined.std(axis=1)
            })

            result.reset_index(inplace=True)  # restore 'Metric' as column

            out_filename = f"{dataset}_{group}_{k}_mean_metrics.csv"
            out_path = os.path.join(method_path, out_filename)
            result.to_csv(out_path, index=False)

    print("Aggregation complete.")

def collect_all_mean_metrics(base_dir="metrics"):
    pattern = re.compile(r"(.+?)_(.+?)_(\d+)_mean_metrics\.csv")
    all_rows = []

    for method in os.listdir(base_dir):
        method_path = os.path.join(base_dir, method)
        if not os.path.isdir(method_path):
            continue

        for file in os.listdir(method_path):
            match = pattern.match(file)
            if not match:
                continue

            dataset, group, k = match.groups()
            file_path = os.path.join(method_path, file)

            try:
                df = pd.read_csv(file_path)
                df["row"] = 0

                # Convert long format to wide format
                wide_df = df.pivot(index="row", columns="Metric", values=["mean", "std"])

                # Flatten multi-level column names like ('mean', 'MRD')
                wide_df.columns = [f"{metric}_{stat}" for stat, metric in wide_df.columns]

                # Add identifying columns
                wide_df['Dataset'] = dataset
                wide_df['group'] = group
                wide_df['k'] = int(k)
                wide_df['method'] = method

                all_rows.append(wide_df)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                return

    if not all_rows:
        return pd.DataFrame()

    df_all = pd.concat(all_rows, ignore_index=True)

    # Reorder columns
    id_cols = ["Dataset", "group", "k", "method"]
    metric_cols = [col for col in df_all.columns if col not in id_cols]
    df_all = df_all[id_cols + sorted(metric_cols)]

    return df_all

def adjust_df(df):
    # Round all metric columns to 4 decimals
    meta_cols = ["Dataset", "group", "k", "method"]
    metric_cols = [col for col in df.columns if col not in meta_cols]
    df.loc[:, metric_cols] = df.loc[:, metric_cols].round(3)

    dataset_to_nr = {
        "Compas,Gender": "D1",
        "Compas,gender": "D1",
        "Compas,Race": "D2",
        "Compas,race": "D2",
        "German,Gender": "D3",
        "German,gender": "D3",
        "German,Age": "D4",
        "German,age": "D4",
        "German,Age,Gender": "D5",
        "German,age,gender": "D5",
        "LSAT,Gender": "D6",
        "LSAT,gender": "D6",
        "LSAT,Race": "D7",
        "LSAT,race": "D7",
        "LSAT,Race,Gender": "D8",
        "LSAT,race,gender": "D8",
    }
    df.loc[:, "Dataset"] = df.loc[:, "Dataset"] + "," + df.loc[:, "group"]
    df.loc[:, "Dataset"] = df.loc[:, "Dataset"].map(dataset_to_nr)

    # Drop original 'group' column
    df = df.drop(columns=["group"])
    df["method"] = df["method"].replace({
        "colorblind": "Colorblind",
        "zehlike": "Multi. Fa*ir",
        "FairExpose-Ord": "FairExpose-Ord",
        "FairExpose-Pro": "FairExpose-Pro",
        "random": "Random",
        "FairNormRank": "FairNormRank",
    })

    # Define your custom order
    method_order = ["Colorblind", "Random", "FairNormRank", "Multi. Fa*ir", "FairExpose-Ord", "FairExpose-Pro"]

    # Convert to categorical with ordered=True
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)

    df = df.sort_values(by=["Dataset", "method"]).reset_index(drop=True)
    df.to_csv("all_results.csv")
    return df


def create_latex_table(df):
    # Combine mean and std
    metric_cols = ['runtime']
    for col in metric_cols:
        df[col] = df.apply(lambda row: f"{row[col + '_mean']:.3f} $\\pm$ {row[col + '_std']:.3f}", axis=1)

    # Keep only desired columns
    final_cols = [
        'Dataset', 'method', 'ED_mean', 'PD_mean', 'RD_mean', 'OD_mean', 'NDCG_mean',
        'kendall_tau_mean', 'runtime'
    ]
    df = df[final_cols]

    datasets_map = {
        "D1": r"\multirow{6}{*}{\begin{tabular}[c]{@{}l@{}} D1 \\ $k=500$ \\ $|\mathcal{G}|=2$\end{tabular}}",
        "D2": r"\multirow{6}{*}{\begin{tabular}[c]{@{}l@{}} D2 \\ $k=90$ \\ $|\mathcal{G}|=9$\end{tabular}}",
        "D3": r"\multirow{5}{*}{\begin{tabular}[c]{@{}l@{}} D3 \\ $k=400$ \\ $|\mathcal{G}|=2$\end{tabular}}",
        "D4": r"\multirow{6}{*}{\begin{tabular}[c]{@{}l@{}} D4 \\ $k=200$ \\ $|\mathcal{G}|=3$\end{tabular}}",
        "D5": r"\multirow{5}{*}{\begin{tabular}[c]{@{}l@{}} D5 \\ $k=200$ \\ $|\mathcal{G}|=6$\end{tabular}}",
        "D6": r"\multirow{5}{*}{\begin{tabular}[c]{@{}l@{}} D6 \\ $k=2\,000$ \\ $|\mathcal{G}|=2$\end{tabular}}",
        "D7": r"\multirow{5}{*}{\begin{tabular}[c]{@{}l@{}} D7 \\ $k=500$ \\ $|\mathcal{G}|=5$\end{tabular}}",
        "D8": r"\multirow{3}{*}{\begin{tabular}[c]{@{}l@{}} D8 \\ $k=300$ \\ $|\mathcal{G}|=10$\end{tabular}}"
    }

    # Rename for LaTeX header formatting
    col_latex_names = {
        'Dataset': 'Dataset',
        'k': 'k',
        'method': 'Method',
        'group_counts_mean' : 'min. Proportion',
        'ED_mean': 'ED',
        'PD_mean': 'PD',
        'MGRD_mean': 'RDg',
        'RD_mean': 'RD',
        'OD_mean': 'OD',
        'kendall_tau_mean': 'Kendall\'s Tau',
        'runtime': 'Runtime',
        'NDCG_mean': 'NDCG'
    }

    # Build LaTeX
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\caption{Experimental results for our proposed methods.}")
    latex.append(r"\label{tab:exp1}")
    latex.append(r"\centering")
    latex.append(r"\begin{tabular}{l l l" + " r " * (len(final_cols) - 3) + r"}")
    latex.append(r"\toprule")
    latex.append(" & ".join([col_latex_names[col] for col in final_cols]) + r" \\")
    latex.append(r"\midrule")

    for dataset, group in df.groupby(['Dataset']):
        group = group.reset_index(drop=True)
        n = len(group)
        for i, row in group.iterrows():
            line = []
            if i == 0:
                line.append(datasets_map[dataset[0]])
            #else:
            #    line.append(rf"& ")
            line.append(f"& {row['method']}")
            for col in final_cols[2:]:  # Skip Dataset, k, method
                val = row[col]
                if col == "group_counts_mean":
                    line.append(f"& {val:.2f}")
                elif isinstance(val, float):
                    line.append(f"& {val:.3f}")
                else:
                    line.append(f"& {val}")
            line.append(r"\\")
            latex.append(" ".join(line))
        latex.append(r"\hline")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")
    latex_code = '\n'.join(latex)

    with open("table_exp2.txt", "w", encoding="utf-8") as f:
        f.write(latex_code)


if __name__ == "__main__":
    aggregate_metrics()
    df = collect_all_mean_metrics()
    df = adjust_df(df)
    create_latex_table(df)
    

import pandas as pd
import glob
import os

def get_baseline_method(method):
    if "OP" in method:
        return "FairOPRankCO"
    elif "PP" in method:
        return "FairPPRankCO"
    else:
        return "FairOPRankCO"

def merge_results(path="metrics/"):
    # Mapping from datatype to group string
    datatype_to_groups = {
        "binary": "a",
        "multi binary": "b",
        "non-binary": "c",
        "multi non-binary": "d"
    }

    # Collect all CSV files
    csv_files = glob.glob("results/metrics/*_metrics.csv")
    dfs = []

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        parts = filename[:-12].split('_')  # remove '_metrics.csv'

        if len(parts) < 4:
            print(f"Skipping malformed filename: {filename}")
            continue

        method, datatype, k, trial = parts

        df = pd.read_csv(file_path, index_col=0)  # index_col=0 handles the saved index
        df_wide = df.set_index("Metric").T
        df_wide["Method"] = method
        df_wide["Datatype"] = datatype  # we'll group by this
        dfs.append(df_wide)

    # Combine all into one dataframe
    full_df = pd.concat(dfs, ignore_index=True)
    print(full_df)

    # Group by method and datatype, compute mean of metric columns
    metrics_cols = [col for col in full_df.columns if col not in ["Method", "Datatype"]]
    print(metrics_cols)
    for col in metrics_cols:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
    grouped = full_df.groupby(["Method", "Datatype"])[metrics_cols].mean().reset_index()

    # Add "groups" column based on datatype
    grouped["groups"] = grouped["Datatype"].map(datatype_to_groups)

    # Reorder columns
    df = grouped[["Method", "Datatype", "groups"] + metrics_cols]

    # calc MPoD loss
    # df['baseline_method'] = df['Method'].apply(get_baseline_method)
    # best_MPOD_df = df[df['Method'].isin(['FairOPRankCO', 'FairPPRankCO'])][['groups', 'Method', 'MPoD']]
    # best_MPOD_df = best_MPOD_df.rename(columns={'Method': 'baseline_method', 'MPoD': 'best_MPOD'})
    # df = df.merge(best_MPOD_df, on=['groups', 'baseline_method'], how='left')
    # df['MPoD loss'] = df['MPoD'] - df['best_MPOD']
    df = df[["Method", "Datatype", "groups", "ED", "PD", "OD", "ODg", "RD", "RDg"]]

    order = [
        "colorblind",
        "FairExpose-Pro",
        "FairExpose-Ord",
        "FairExpose-Ord Greedy",
    ]
    # Convert Method column to a categorical type with the given order
    df['Method'] = pd.Categorical(df['Method'], categories=order, ordered=True)
    df = df.sort_values(by=["groups", "Method"]).reset_index(drop=True)

    metrics_cols = [col for col in df.columns if col not in ["Method", "groups"]]
    df[metrics_cols] = df[metrics_cols].round(3) # Round metric columns to 3 decimals
    df = df.rename(columns={'Datatype': 'Prot. Attr.'})

    # add columns for Prot. Attr.
    group_map = {
        "binary": 1,
        "multi binary": 2,
        "non-binary": 1,
        "multi non-binary": 2
    }
    df[r"\#Attr."] = df['Prot. Attr.'].map(group_map)
    group_map = {
        "binary": "2",
        "multi binary": "4",
        "non-binary": "4",
        "multi non-binary": "9"
    }
    df[r"$|\mathcal{G}|$"] = df['Prot. Attr.'].map(group_map)
    # df = df.drop("GOCD", axis=1)

    df.to_csv("results/merged_df.csv")
    return df


def df_to_latex_multirow(df, col_order=None, caption=None, label=None, min_multirow_len=3):
    if col_order is None:
        col_order = df.columns.tolist()

    def multirowify_runs(series, min_len=3):
        """
        Given a pd.Series, find consecutive runs of identical values of length >= min_len.
        Return a list with multirow applied to first value of each run and '' elsewhere.
        """
        vals = series.values
        result = [''] * len(series)

        start_idx = 0
        while start_idx < len(vals):
            # Find run length
            run_val = vals[start_idx]
            run_end = start_idx + 1
            while run_end < len(vals) and vals[run_end] == run_val:
                run_end += 1
            run_len = run_end - start_idx

            if run_len >= min_len:
                # multirow command for first row of run
                result[start_idx] = f"\\multirow{{{run_len}}}{{*}}{{{vals[start_idx]}}}"
                # rest are empty strings already
            else:
                # for shorter runs, just copy values as strings
                for i in range(start_idx, run_end):
                    result[i] = str(vals[i])

            start_idx = run_end
        return result

    # Multirow for "groups" column over all rows (consecutive repeats)
    def multirowify_groups(col_values):
        result = []
        prev = None
        count = 0
        for i, val in enumerate(col_values):
            if val == prev:
                count += 1
            else:
                if count > 1:
                    start = i - count
                    result[start] = f"\\multirow{{{count}}}{{*}}{{{result[start]}}}"
                    for j in range(start + 1, i):
                        result[j] = ''
                count = 1
                prev = val
            result.append(str(val) if i >= len(result) else result[i])

        if count > 1:
            start = len(col_values) - count
            result[start] = f"\\multirow{{{count}}}{{*}}{{{result[start]}}}"
            for j in range(start + 1, len(col_values)):
                result[j] = ''
        if not result:
            result = col_values.astype(str).tolist()
        return result

    # Prepare output data
    data = pd.DataFrame(index=df.index)

    # Apply multirow for groups column
    data["groups"] = multirowify_groups(df["groups"])

    # Copy Method column as strings
    if "Method" in df.columns:
        data["Method"] = df["Method"].astype(str)

    # For all other metric columns: apply multirow runs within each group separately
    metric_cols = [c for c in col_order if c not in ["groups", "Method"]]

    for col in metric_cols:
        multirowed_col = []
        # Split df by groups to respect group boundaries
        for group_val, group_df in df.groupby("groups", sort=False):
            series = group_df[col]
            multirowed_part = multirowify_runs(series, min_len=min_multirow_len)
            multirowed_col.extend(multirowed_part)
        data[col] = multirowed_col

    # Construct LaTeX lines
    lines = []
    lines.append(r"\begin{table*}[t]")
    if caption:
        lines.append(f"    \\caption{{{caption}}}")
    if label:
        lines.append(f"    \\label{{{label}}}")
    lines.append("    \\centering")

    # Align: left for text, right for numbers
    aligns = []
    for col in col_order:
        if pd.api.types.is_numeric_dtype(df[col]):
            aligns.append('r')
        else:
            aligns.append('l')
    col_def = " ".join(aligns)
    lines.append(f"    \\begin{{tabular}}{{{col_def}}}")
    lines.append("    \\toprule")

    header = " & ".join(col_order) + r" \\"
    lines.append(f"    {header}")
    lines.append("    \\midrule")

    for i in range(len(df)):
        row = []
        for col in col_order:
            cell = data[col].iloc[i]
            row.append(cell if cell != '' else '')  # empty string for repeated cells
        lines.append("    " + " & ".join(row) + r" \\")
        # Add \hline after every 3 rows except after last row
        if (i + 1) % 4 == 0 and i != len(df) - 1:
            lines.append("    \\hline")

    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


if __name__ == "__main__":
    df = merge_results()
    df = df[df["Method"].isin(
        ["colorblind", "FairExpose-Pro", "FairExpose-Ord","FairExpose-Ord Greedy"])]

    latex_code = df_to_latex_multirow(
        df,
        col_order=["Prot. Attr.", r"\#Attr.", r"$|\mathcal{G}|$", "groups", "Method"] +
                  [col for col in df.columns if col not in ["Prot. Attr.", r"\#Attr.", r"$|\mathcal{G}|$", "groups", "Method"]],
        caption="Experimental results for out proposed methods on generated datasets.",
        label="tab:exp1"
    )
    with open("results/table_exp1.tex", "w", encoding="utf-8") as f:
        f.write(latex_code)
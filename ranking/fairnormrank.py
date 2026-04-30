import math
import pandas as pd


def fair_norm_rank(data, k, target_proportions, group_col, relevance_col):
    """
    Selects the top-k candidates in a fair way based on proportional fairness (FairNormRank algorithm).

    Parameters:
        problem (Problem): the problem to be ranked with the candidates, k, and target proportions.

    Returns:
        pd.DataFrame: A fair top-k ranking according to proportional fairness of candidates.
    """
    # check target_proportion
    total = sum(target_proportions.values())
    zeros = [k for k, v in target_proportions.items() if v == 0]
    groups = data[group_col].unique()

    if zeros:
        remaining = 1 - total
        share = remaining / len(zeros)

        for key in zeros:
            target_proportions[key] = share

    # Create a ranking for each group separately by quality score
    df = data.sort_values(by=relevance_col, ascending=False).reset_index(drop=True)
    group_rankings = {group: df[df[group_col] == group][:k].to_dict('records') for group in groups}

    # Select top candidates from each group according to target proportions
    top_ranking = []
    residual_x = []
    for g in groups:
        n_group = math.floor(target_proportions[g] * k)
        if n_group > 0:
            top_ranking += group_rankings[g][:n_group]
            residual_x += [group_rankings[g][n_group]]  # store next best candidate from each group

    # Select k-best candidates using top_ranking and residual_x
    residual_x.sort(key=lambda d: d[relevance_col], reverse=True)
    top_k_ranking = top_ranking + residual_x[:k - len(top_ranking)]   # compensate rounding
    top_k_ranking.sort(key=lambda d: d[relevance_col], reverse=True)
    top_k_ranking = pd.DataFrame(top_k_ranking).reset_index(drop=True)

    return top_k_ranking

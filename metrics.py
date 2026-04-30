import numpy as np
from scipy.stats import kendalltau


def calc_relevance_disparity(df, top_K, relevance_score="y"):
    """
    Computes the relevance disparity score of a ranking.

    Parameters:
        df (pd.DataFrame): DataFrame containing all candidates.
        top_K (pd.DataFrame): Dataframe containing the ordered top-k candidates.
        relevance_score (str): Column name containing quality scores.

    Returns:
        float: The relevance disparity of the ranking.
    """
    df_not_topk = df.loc[~df["id"].isin(top_K["id"])]

    max_relevance_not_topk = np.max(df_not_topk[relevance_score])
    min_relevance_topk = np.min(top_K[relevance_score])

    if max_relevance_not_topk > min_relevance_topk:
        rd_raw = max_relevance_not_topk - min_relevance_topk
        rd = rd_raw / (max(df[relevance_score].values) - min(df[relevance_score].values))
    else:
        rd = 0

    return rd

def calc_group_relevance_disparity(df, top_k, protected_attr="z", relevance_score="y"):
    """
    Computes the group relevance disparity of a ranking defined as the maximum groupwise selection disparity.

    Parameters:
        df (pd.DataFrame): DataFrame containing all candidates.
        top_k (pd.DataFrame): Dataframe containing the ordered top-k candidates.
        protected_attr (str): Column name representing the protected group attribute.
        relevance_score (str): Column name containing quality scores.

    Returns:
        float: The group relevance disparity of the ranking.
    """
    relevance_disparities = []

    # calculate groupwise selection disparities
    for group in set(top_k[protected_attr]):
        top_k_g = top_k[top_k[protected_attr] == group]
        df_g = df[df[protected_attr] == group]

        rd = calc_relevance_disparity(df_g, top_k_g, relevance_score)
        relevance_disparities.append(rd)

    return max(relevance_disparities)

def calc_ordering_disparity(top_k, relevance_score="y"):
    """
    Compute the Ordering Disparity (OD) for a top-k ranking.

    Parameters:
        top_k (pd.DataFrame): Dataframe containing the ordered top-k candidates.
        relevance_score (str): Column name containing quality scores.

    Returns:
        float: Ordering Disparity of the ranking.
    """
    if len(top_k) < 2:
        return 0.0

    max_gap = 0.0
    scores = top_k[relevance_score].values
    n = len(scores)

    for i in range(n-1):
        for j in range(i+1, n):
            # Candidate i is ranked before j (i < j), but has lower relevance
            if scores[i] < scores[j]:
                gap = scores[j] - scores[i]
                max_gap = max(max_gap, gap)

    norm = scores.max() - scores.min()

    return max_gap / norm if norm > 0 else 0

def calc_group_ordering_disparity(top_k, protected_attr="z", relevance_score="y"):
    """
    Compute the Group Ordering Disparity (ODg) for a top-k ranking.

    Parameters:
        top_k (pd.DataFrame): Dataframe containing the ordered top-k candidates.
        protected_attr (str): Column name representing the protected group attribute.
        relevance_score (str): Column name containing quality scores.

    Returns:
        float: Ordering Disparity of the ranking.
    """
    ordering_disparities = []

    for group in set(top_k[protected_attr]):
        top_k_g = top_k[top_k[protected_attr] == group]

        od = calc_ordering_disparity(top_k_g, relevance_score)
        ordering_disparities.append(od)

    # odgs = top_k.groupby(protected_attr).apply(lambda df: calc_ordering_disparity(df, relevance_score=relevance_score), include_groups=False)
    return max(ordering_disparities)

def calc_proportion_disparity(top_k, target_proportions, protected_attr="z"):
    """
    Calculates the Proportion Disparity for the top-k candidates. This metric is normalized to the
    maximum possible Proportion Disparity based on the target proportions.

    Parameters:
        top_k (pd.DataFrame): The ranking containing candidates, groups, and quality scores.
        protected_attr (str): The column of the social groups.
        target_proportions (dict): A dictionary mapping group labels to their target proportion in top-k.

    Returns:
        float: The Proportion Disparity across all groups.
    """
    # Get observed proportions of groups in the top-k ranking
    observed_proportions = top_k[protected_attr].value_counts(normalize=True).to_dict()

    # Ensure all groups are accounted for, even if they have 0 representation
    observed_proportions = {group: observed_proportions.get(group, 0) for group in target_proportions}

    disparities = []
    for group in observed_proportions:
        disparity = target_proportions[group] - observed_proportions[group]
        if disparity > 0:
            disparities.append(disparity / target_proportions[group])

    pd = 0
    if len(disparities) > 0:
        pd = max(disparities)

    return pd

def calc_ndcg(top_k, relevance_col="y"):
    """
    Computes the Normalized Discounted Cumulative Gain (NDCG) score.

    Parameters:
        top_k (pd.DataFrame): The ranking containing candidates, groups, and quality scores.
        relevance_col (str): Column name containing quality scores.

    Returns:
        float: NDCG score between 0 and 1.
    """
    # Compute DCG
    ranked_scores = np.array(top_k[relevance_col].values)
    if len(ranked_scores) > 0:
        dcg = np.sum(ranked_scores / np.log2(np.arange(1, len(ranked_scores) + 1) + 1))
    else:
        return np.nan

    # Compute Ideal DCG (IDCG) with perfectly sorted ranking
    ideal_scores = np.sort(ranked_scores)[::-1]
    idcg = np.sum(ideal_scores / np.log2(np.arange(1, len(ideal_scores) + 1) + 1))

    # Normalize: If IDCG is 0, return 0 to avoid division by zero
    return dcg / idcg if idcg > 0 else 0.0

def calc_exposure_disparity(top_k, groups, protected_attr="z", weights=None):
    """
        Computes the Exposure Disparity (ED) for a ranked list.

        Parameters:
            top_k (pd.DataFrame): A DataFrame containing ranked candidates.
            groups (list): A list of all social groups.
            protected_attr (str): The column name representing the protected group attribute.

        Returns:
            float: A normalized positional fairness score between 0 and 1.
    """
    if not weights:
        positions = np.arange(1, len(top_k) + 1)
        weights = 1 / np.log2(positions + 1)

    # Extract group labels as a NumPy array
    group_labels = top_k[protected_attr].to_numpy()

    # Pre-allocate a dict for group scores
    group_scores = {group: 0.0 for group in groups}

    for group in groups:
        mask = group_labels == group
        if np.any(mask):
            group_scores[group] = weights[mask].sum()

    ed = max(group_scores.values()) - min(group_scores.values())
    # norm = weights.sum()
    return ed

def calc_kendall_tau(df, top_k, relevance_score="y"):
    """
    Computes the kendall tau between the top-k ranking and colorblind relevanced-based ranking.

    Parameters:
        df (pd.DataFrame): DataFrame containing all candidates.
        top_k (pd.DataFrame): Dataframe containing the ordered top-k candidates.
        relevance_score (str): Column name containing quality scores.

    Returns:
        float: kendall tau between -1 and 1, where 1=perfect correlation.
    """
    k = len(top_k)
    # Top-k reference colorblind ranking
    rank_orig = (df.sort_values(by=relevance_score, ascending=False).head(k).reset_index(drop=True))

    # Create rank dictionaries
    orig_rank_dict = {row["id"]: i + 1 for i, row in rank_orig.iterrows()}
    fair_rank_dict = {row["id"]: i + 1 for i, row in top_k.iterrows()}

    # Union of ids
    all_ids = set(orig_rank_dict.keys()).union(fair_rank_dict.keys())

    # Assign missing items rank = k+1
    orig_ranks = []
    fair_ranks = []

    for _id in all_ids:
        orig_ranks.append(orig_rank_dict.get(_id, k + 1))
        fair_ranks.append(fair_rank_dict.get(_id, k + 1))

    kendall_corr, _ = kendalltau(orig_ranks, fair_ranks)
    return kendall_corr

def get_metrics(problem, rank_df):
    data = problem.candidates
    # Utility
    rd = calc_relevance_disparity(data, rank_df, relevance_score=problem.relevance_col)
    rdg = calc_group_relevance_disparity(data, rank_df, problem.group_col, relevance_score=problem.relevance_col)
    od = calc_ordering_disparity(rank_df, relevance_score=problem.relevance_col)
    odg = calc_group_ordering_disparity(rank_df, relevance_score=problem.relevance_col)

    # Fairness
    pd = calc_proportion_disparity(rank_df, problem.proportional_target, problem.group_col)
    ed = calc_exposure_disparity(rank_df, data[problem.group_col].unique(), protected_attr=problem.group_col)

    # Others
    ndcg = calc_ndcg(rank_df, relevance_col=problem.relevance_col)
    kendall_tau = calc_kendall_tau(data, rank_df, relevance_score=problem.relevance_col)


    metrics = {"RD": rd,
               "RDg": rdg,
               "OD": od,
               "ODg": odg,
               "PD": pd,
               "ED": ed,
               "NDCG": ndcg,
               "kendall_tau": kendall_tau}

    return metrics

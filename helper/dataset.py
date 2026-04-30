import numpy as np
import pandas as pd


def partial_normalization(df, relevance_score="y", protected_attr="z", alpha=1.0):
    """
    Applies partial normalization to the "y" column within each group in "z".

    Parameters:
        df (pd.DataFrame): The input dataframe.
        relevance_score: column name of the relevance score column.
        alpha (float): Degree of normalization (0 = no normalization, 1 = full normalization).
    Returns:
        pd.DataFrame: DataFrame with a new column "y_normalized".
    """
    def normalize_with_alpha(x, alpha):
        min_x, max_x = x.min(), x.max()
        if max_x == min_x:  # Avoid division by zero
            return x
        normalized = (x - min_x) / (max_x - min_x)  # Standard min-max normalization
        return alpha * normalized + (1 - alpha) * x  # Interpolate with original values

    df[relevance_score + "_norm"] = (df.groupby(protected_attr)[relevance_score]
                             .transform(lambda x: normalize_with_alpha(x, alpha)))
    return df

def generate_dataset(n, protected_attributes, unfairness=0.5, seed=40):
    """
    Generates a dataset for the fair ranking problem with multiple protected attributes.

    Parameters:
        n (int): Number of candidates.
        protected_attributes (dict): A dictionary where keys are attribute names and values are the number of groups
        (e.g. {"gender": 2, "ethnicity": 3}).
        unfairness (float): Degree of unfairness (0 = fair, higher values increase bias).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: A dataset containing a "z" column with all protected attributes as a string,
                      and a "y" column representing the relevance score.
    """
    if seed is not None:
        np.random.seed(seed)

    relevance_score = "y"
    protected_attr = "z"

    # Generate group assignments for each protected attribute
    groups = {attr: np.random.choice(range(num_groups), size=n) for attr, num_groups in protected_attributes.items()}

    # Create a combined group column (as a string with "_")
    df = pd.DataFrame(groups)
    df[protected_attr] = df.astype(str).agg("_".join, axis=1)

    # Define base quality scores from a normal distribution
    base_quality = np.random.normal(loc=0.5, scale=0.15, size=n)

    # Introduce unfairness based on multiple attributes
    unfairness_effect = np.zeros(n)
    for attr, num_groups in protected_attributes.items():
        unfairness_effect += (df[attr] / (num_groups - 1)) * (unfairness / 2)

    quality_scores = np.clip(base_quality + unfairness_effect, 0, 1)  # Ensure scores stay in [0,1]

    # Assign quality score
    df[relevance_score] = quality_scores

    # Sort by quality score (descending) for ranking
    df = df.sort_values(by=relevance_score, ascending=False).reset_index(drop=True)
    df["id"] = df.index  # Assign ranking ID
    df[relevance_score] = np.linspace(1, 0, len(df))
    return df, relevance_score, protected_attr

def load_data(dataset_name, protected):
    if dataset_name == "Compas":
        use_cols = ['Ethnic_Code_Text', 'Sex_Code_Text', 'DateOfBirth', 'RawScore', 'DisplayText']
        data = pd.read_csv(
            "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-raw.csv",
            usecols=use_cols)

        # Drop rows with missing values
        data = data.dropna(axis=0, how='any')

        # select Risk of Recidivism scores
        data = data[data["DisplayText"] == "Risk of Recidivism"]
        data["Ethnic_Code_Text"] = data["Ethnic_Code_Text"].replace("African-Am", "African-American")

        # combine protected attributes to new column
        relevance_score = 'RawScore'

        # protected_attributes = ["Sex_Code_Text", "Ethnic_Code_Text"]  # caution: one group has only 1 member
        if protected == "Race":
            protected_attributes = ["Ethnic_Code_Text"]
        elif protected == "Gender":
            protected_attributes = ["Sex_Code_Text"]
        elif protected == "Race,Gender":
            protected_attributes = ["Ethnic_Code_Text", "Sex_Code_Text"]

        data["z"] = data[protected_attributes].astype(str).agg("_".join, axis=1)
        protected_col = "z"

        # reverse Score
        data[relevance_score] *= -1

        data = data.drop('DisplayText', axis=1)
        data["id"] = data.index
    elif dataset_name == "German":
        file = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

        names = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount',
                 'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors',
                 'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing',
                 'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']

        data = pd.read_csv(file, delimiter=' ', names=names)

        employment_map = {
            'A71': 0.0,
            'A72': 0.5,
            'A73': 2.5,
            'A74': 5.5,
            'A75': 8.0
        }

        # Apply the mapping
        data['employment_years'] = data['employmentsince'].map(employment_map)

        # Manual min-max normalization
        for col in ['duration', 'creditamount', 'employment_years']:
            min_val = data[col].min()
            max_val = data[col].max()
            data[col] = (data[col] - min_val) / (max_val - min_val)

        # Map to simplified 'Gender' column
        def extract_gender(val):
            if val in ['A91', 'A93', 'A94']:
                return 'male'
            elif val in ['A92', 'A95']:
                return 'female'
            else:
                return 'unknown'

        # Define the age categories
        def categorize_age(age):
            if age < 25:
                return 'young'
            elif 25 <= age <= 45:
                return 'middle'
            else:
                return 'old'

        # Apply the function to create the new column
        data['age_cat'] = data['age'].apply(categorize_age)

        data["gender"] = data["statussex"].apply(extract_gender)
        data["quality"] = 1 / 3 * data["duration"] + 1 / 3 * data["creditamount"] + 1 / 3 * data["employment_years"]
        data = data[["gender", 'age_cat', 'quality']].copy()
        data["id"] = data.index
        relevance_score = "quality"

        # protected_attributes = ["Sex_Code_Text", "Ethnic_Code_Text"]  # caution: one group has only 1 member
        if protected == "Age":
            protected_attributes = ["age_cat"]
        elif protected == "Gender":
            protected_attributes = ["gender"]
        elif protected == "Age,Gender":
            protected_attributes = ["age_cat", "gender"]

        data["z"] = data[protected_attributes].astype(str).agg("_".join, axis=1)
        protected_col = "z"

    elif dataset_name == "LSAT":
        try:
            data = pd.read_csv("helper/data/bar_pass_prediction.csv")
        except:
            data = pd.read_csv("../../../helper/data/bar_pass_prediction.csv")
        data = data[["lsat", 'gender', 'race1']].copy()
        relevance_score = "lsat"

        if protected == "Gender":
            protected_attributes = ["gender"]
        elif protected == "Race":
            protected_attributes = ["race1"]
        elif protected == "Race,Gender":
            protected_attributes = ["race1", "gender"]

        data = data.dropna()
        data["z"] = data[protected_attributes].astype(str).agg("_".join, axis=1)
        protected_col = "z"
        data["id"] = data.index


    elif dataset_name == "generated":
        if protected == "binary":
            protected_attributes = {"a": 2}
        elif protected == "non-binary":
            protected_attributes = {"a": 4}
        elif protected == "multi binary":
            protected_attributes = {"a": 2, "b": 2}
        elif protected == "multi non-binary":
            protected_attributes = {"a": 3, "b": 2}

        data, relevance_score, protected_col = generate_dataset(n=100000, protected_attributes=protected_attributes, unfairness=0.5, seed=42)  # 100000
    else:
        print("Unknown dataset name")

    # normalize scores between 0 and 1
    data["quality"] = (data[relevance_score] - data[relevance_score].min()) / (
                data[relevance_score].max() - data[relevance_score].min())
    relevance_score = "quality"
    groups = len(data["z"].unique())
    print(f"Relevance Score: {relevance_score}, Protected Attributes: {protected_attributes}, Groups: {groups}")

    return data, relevance_score, protected_col

# Exact Optimization for Exposure Fairness in Top-*k* Rankings

This repository implements FairExpose, a n optimization framework for fairness-aware top-k ranking of the paper "Exact Optimization for Exposure Fairness in Top-*k* Rankings".
The method optimizes exposure fairness to ensure that social groups receive comparable levels of user attention in a ranking.

## Example
```python
from ranking import RankingProblem
from helper import load_data
from ranking import FairExpose
from metrics import *

# Load Data with multiple non-binary protected groups
data, relevance_col, group_col = load_data(dataset_name="German", protected="Gender")

# Create top-k ranking problem with "females" as protected group
problem = RankingProblem(data, group_col, relevance_col, k=30, proportional_target={"female": 0.5})

# Compute fair top-k ranking
ranker = FairExpose(problem)
ranker.fair_expose_pro()
top_k = ranker.topk

# Compute evaluation metrics of the resulting top-k ranking
metrics = get_metrics(problem, ranker.topk)
print(metrics)
# >>> 'RD': 0.159, 'RDg': 0.0, 'OD': 0.585, 'ODg': 0.0, 'PD': 0.0, 'ED': 0.0002, 'NDCG': 0.982, 'kendall_tau': 0.271
```

## Project Structure
- **ranking/**  
  Formulation and algorithms for the fair top-k ranking problem with optimal exposure fairness.
- **metrics**  
  computation of utility and fairness metrics.
- **helper/**  
  Dataset loaders and preprocessing utilities used in experiments.
- **experiments/**
  Experimental pipelines for reproducing the results in the paper, as well as all experimental results.

## Requirements
This framework was tested with Python 3.11.2 and relies on:
- `numpy`
- `pandas`
- `ortools`
- `scipy`

All dependencies are listed in `requirements.txt`.

## Project Setup
```bash
git clone <FairExpose>
cd FairExpose
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

import pandas as pd
import math
from collections import defaultdict
from ortools.sat.python import cp_model
import numpy as np

class FairExpose(object):
    def __init__(self, problem):
        """
        A class to compute a fair top-k ranking of a given ranking problem with optimal exposure fairness.

        Attributes
        ----------
        problem : Problem
            The problem to compute the fair top-k ranking for.
        algorithm : Algorithm
            The algorithm to use to compute the fair top-k ranking.
        """
        self.problem = problem
        self.data = self.problem.candidates
        self.weights = [1/np.log2(x+1) for x in range(1, self.problem.k+1)]
        self.topk = None

    def fair_expose_ord_greedy(self, ):
        """
        A method FairExpose-Ord Greedy to compute a fair top-k ranking with near-optimal exposure fairness and optimal
        ordering utility, as well as group selection utility.
        """
        # preprocess
        df = (self.data.sort_values(by=self.problem.relevance_col, ascending=False)
              .groupby(self.problem.group_col, group_keys=False)
              .head(self.problem.k)
              .reset_index(drop=True))

        G = len(self.problem.groups)
        target = sum(self.weights) / G
        pfs = dict.fromkeys(self.problem.groups, 0.0)

        # remaining counts
        remaining = (
            pd.get_dummies(df[self.problem.group_col])
            .iloc[::-1]
            .cumsum()
            .iloc[::-1]
            .to_dict("records")
        )
        ranking = []
        rejected = set()

        for pointer, row in df.iterrows():
            if len(ranking) == self.problem.k:
                break

            g = row[self.problem.group_col]
            if g in rejected:
                continue

            pos = len(ranking)
            gain = self.weights[pos]
            before = abs(pfs[g] - target)
            after = abs(pfs[g] + gain - target)

            remaining_slots = self.problem.k - pos

            # feasibility check
            available = sum(
                remaining[pointer][h]
                for h in self.problem.groups
                if h not in rejected and h != g
            )
            can_reject = available >= remaining_slots

            # decision to include candidate or skip
            if after < before or len(rejected) == G - 1 or not can_reject:
                ranking.append(row)
                pfs[g] += gain
            else:
                rejected.add(g)

        self.topk = pd.DataFrame(ranking).reset_index(drop=True)


    def fair_expose_ord(self,):
        """
        A method FairExpose-Ord to compute a fair top-k ranking with optimal exposure fairness and ordering
        utility, as well as group selection utility.
        """
        # preprocessing
        df = (self.data.sort_values(by=self.problem.relevance_col, ascending=False)
              .groupby(self.problem.group_col, group_keys=False)
              .head(self.problem.k)
              .reset_index(drop=True))
        df["rank"] = np.arange(len(df))

        groups = list(df[self.problem.group_col].unique())
        G = len(groups)
        g2i = {g: i for i, g in enumerate(groups)}
        i2g = {i: g for g, i in g2i.items()}
        group_ids = df[self.problem.group_col].map(g2i).to_numpy()
        group_ranks = df.groupby(self.problem.group_col).cumcount().to_numpy()
        n = len(df)
        desired_pf = sum(self.weights) / G

        # greedy upper bound
        self.fair_expose_ord_greedy()
        greedy_pfs = np.zeros(G)
        for i, row in self.topk.iterrows():
            greedy_pfs[g2i[row[self.problem.group_col]]] += self.weights[i]
        self.topk = None

        best_gap = greedy_pfs.max() - greedy_pfs.min()
        seen_counts = np.zeros((n, G), dtype=np.int16)
        counts = np.zeros(G, dtype=np.int16)
        for i in range(n):
            gi = group_ids[i]
            counts = counts.copy()
            counts[gi] += 1
            seen_counts[i] = counts
        zero_seen = np.zeros(G, dtype=np.int16)

        # DP
        DP = defaultdict(list)
        init_state = tuple([0] * G)
        DP[(init_state, 0)] = [np.zeros(G)]
        best_solution = None

        while DP:
            new_DP = defaultdict(list)
            for (state_tuple, pointer), pfs_list in DP.items():
                state = np.array(state_tuple, dtype=np.int16)
                for pfs in pfs_list:
                    pos = state.sum()

                    # finished ranking
                    if pos == self.problem.k:
                        gap = pfs.max() - pfs.min()
                        if gap <= best_gap + 1e-6:
                            best_gap = gap
                            best_solution = (state_tuple, pfs.copy())
                        continue

                    if pointer >= n:
                        continue
                    if pos + (n - pointer) < self.problem.k:
                        continue

                    seen = seen_counts[pointer - 1] if pointer > 0 else zero_seen
                    non_usable_mask = state < seen
                    gaps = np.abs(pfs - desired_pf)

                    # pruning overshoot
                    if np.any((pfs > desired_pf) & (gaps > best_gap)):
                        continue
                    # pruning undershoot and non-usable
                    if np.any((pfs < desired_pf) & (gaps > best_gap) & non_usable_mask):
                        continue

                    # current candidate
                    gi = group_ids[pointer]
                    group_rank = group_ranks[pointer]
                    gain = self.weights[pos]

                    # take candidate
                    if group_rank == state[gi]:
                        new_state = state.copy()
                        new_state[gi] += 1
                        new_state_tuple = tuple(new_state)

                        new_pfs = pfs.copy()
                        new_pfs[gi] += gain

                        new_DP[(new_state_tuple, pointer + 1)].append(new_pfs)

                    # skip candidate
                    new_DP[(state_tuple, pointer + 1)].append(pfs.copy())

            DP = new_DP

        # reconstruct solution
        if best_solution is None:
            return pd.DataFrame()

        best_state, _ = best_solution

        selected = []
        for gi, count in enumerate(best_state):
            g = i2g[gi]
            selected.append(df[df[self.problem.group_col] == g].head(count))

        result = pd.concat(selected).sort_values(by="rank", ascending=True)
        self.topk = result.reset_index(drop=True)


    def _subset_sum_solver_varying(self, subset_sizes, targets, time_limit=None):
        n = len(self.weights)
        num_subsets = len(self.problem.groups)
        # Scale floats to integers for CP-SAT
        scale = 10_000
        w_int = [int(round(w * scale)) for w in self.weights]
        T_ints = [int(round(t * scale)) for t in targets]
        max_weight_sum = sum(w_int)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8  # use multiple threads
        solver.parameters.cp_model_presolve = True  # enable presolve
        solver.parameters.linearization_level = 1  # faster linearization
        if time_limit:
            solver.parameters.max_time_in_seconds = time_limit
        model = cp_model.CpModel()

        # Assignment matrix: assign[i,s] = 1 if item i is in subset s
        assign = {}
        for i in range(n):
            for s in range(num_subsets):
                assign[i, s] = model.NewBoolVar(f'assign_{i}_{s}')


        for s in range(num_subsets):
            model.Add(sum(assign[i, s] for i in range(n)) == subset_sizes[s])

        # Each item assigned to exactly one subset
        for i in range(n):
            model.Add(sum(assign[i, s] for s in range(num_subsets)) == 1)

        # Subset sums and differences from targets
        diffs = []
        for s in range(num_subsets):
            subset_sum = model.NewIntVar(0, max_weight_sum, f'subset_sum_{s}')
            model.Add(subset_sum == sum(assign[i, s] * w_int[i] for i in range(n)))

            diff = model.NewIntVar(0, max_weight_sum, f'diff_{s}')
            model.AddAbsEquality(diff, subset_sum - T_ints[s])
            diffs.append(diff)

        max_diff = model.NewIntVar(0, max_weight_sum, "max_diff")
        for d in diffs:
            model.Add(d <= max_diff)

        model.Minimize(max_diff)

        # Solve
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            print("Solver status: OPTIMAL")
        elif status == cp_model.FEASIBLE:
            print("Solver status: FEASIBLE (may not be optimal)")
        elif status == cp_model.INFEASIBLE:
            print("Solver status: INFEASIBLE")
        elif status == cp_model.MODEL_INVALID:
            print("Solver status: MODEL_INVALID")
        else:
            print("Solver status: UNKNOWN")

        # Extract solution
        result = []
        for s in range(num_subsets):
            indices = [i for i in range(n) if solver.BooleanValue(assign[i, s])]
            subset_sum = sum(self.weights[i] for i in indices)
            result.append({
                "indices": indices,
                "sum": subset_sum,
            })

        return result

    def fair_expose_pro(self, time_limit=None):
        """
        A method FairExpose-Pro to compute a fair top-k ranking with optimal exposure fairness and proportional
        fairness, as well as group selection utility.
        """
        # preprocess
        data = self.data.sort_values(by=self.problem.relevance_col,
                                             ascending=False).reset_index(drop=True).copy()
        group_sizes = self._calc_target_group_sizes_rel()
        z_unique = data[self.problem.group_col].unique()
        pos_targets = [sum(self.weights) / len(z_unique) for i in range(len(z_unique))]

        # solve for positions
        result = self._subset_sum_solver_varying(
            list(group_sizes.values()),
            pos_targets,
            time_limit
        )

        # construct top-k from group positions
        group_buckets = {g: df.reset_index(drop=True) for g, df in data.groupby(self.problem.group_col, sort=False)}
        output_df = pd.DataFrame(index=range(self.problem.k), columns=data.columns)
        used_groups = set()

        for r in result:
            indices, size = r['indices'], len(r['indices'])
            group_name = None
            for g, t in group_sizes.items():
                if group_sizes.get(g) == size and g not in used_groups:
                    group_name = g
                    break

            used_groups.add(group_name)
            output_df.iloc[indices, :] = group_buckets[group_name].iloc[:size].values

        self.topk = output_df

    def _calc_target_group_sizes_rel(self, ):
        """
        Compute target sizes per group given protected proportions, then fill remaining
        slots with most relevant items (regardless of protection).
        """
        # compute remaining slots
        target_sizes = {key: math.ceil(v * self.problem.k) for key, v in self.problem.proportional_target.items()}
        if sum(target_sizes.values()) > self.problem.k:
            target_sizes = {key: math.floor(v * self.problem.k) for key, v in self.problem.proportional_target.items()}
        diff = self.problem.k - sum(target_sizes.values())
        if diff <= 0:
            return target_sizes

        data_sorted = self.data.sort_values(by=self.problem.relevance_col, ascending=False).copy()

        def mark_assigned(group):
            size = target_sizes.get(group.name, 0)
            group[self.problem.group_col] = group.name
            group = group.copy()
            group['assigned'] = False
            if size > 0:
                group.iloc[:size, group.columns.get_loc('assigned')] = True
            return group

        data_sorted = data_sorted.groupby(self.problem.group_col, group_keys=False).apply(mark_assigned)

        remaining = data_sorted[~data_sorted['assigned']].head(diff)
        remaining_counts = remaining.groupby(self.problem.group_col).size()
        for g, cnt in remaining_counts.items():
            target_sizes[g] = target_sizes.get(g, 0) + cnt

        return target_sizes



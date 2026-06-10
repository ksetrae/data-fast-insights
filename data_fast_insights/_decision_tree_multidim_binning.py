import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, _tree


from collections import defaultdict

def reduce_overlapping_segments(raw_segments: list) -> list:
    """
    - Squashes nested continuous bounds (e.g., '<= 5.04' AND '<= 3.07' becomes '<= 3.07').
    - Consolidates nested one-hot inclusions into unified distinct structures.
    """
    unique_segments = []
    seen_representations = set()

    for seg in raw_segments:
        # 1. Squash Numerical Overlaps
        num_bounds = defaultdict(lambda: {"min": -np.inf, "max": np.inf})
        for col, op, val in seg["num"]:
            if op == "<=":
                num_bounds[col]["max"] = min(num_bounds[col]["max"], val)
            elif op == ">":
                num_bounds[col]["min"] = max(num_bounds[col]["min"], val)

        clean_num = []
        for col, bounds in num_bounds.items():
            if bounds["min"] != -np.inf:
                clean_num.append((col, ">", bounds["min"]))
            if bounds["max"] != np.inf:
                clean_num.append((col, "<=", bounds["max"]))

        # 2. Squash Categorical Overlaps
        cat_inc_map = defaultdict(set)
        for col, val in seg["cat_include"]:
            cat_inc_map[col].add(val)

        cat_exc_map = defaultdict(set)
        for col, val in seg["cat_exclude"]:
            cat_exc_map[col].add(val)

        # Re-format back to unified clean tuples
        clean_inc = [(col, sorted(list(vals))) for col, vals in cat_inc_map.items()]
        clean_exc = [(col, sorted(list(vals))) for col, vals in cat_exc_map.items()]

        # 3. Structural De-duplication Check
        # Convert our cleaned properties to an immutable string tuple key to ensure strict unique outputs
        struct_fingerprint = (
            tuple(sorted(clean_num)),
            tuple(sorted((c, tuple(v)) for c, v in clean_inc)),
            tuple(sorted((c, tuple(v)) for c, v in clean_exc))
        )

        if struct_fingerprint not in seen_representations:
            seen_representations.add(struct_fingerprint)
            unique_segments.append({
                "num": clean_num,
                "cat_include": clean_inc,
                "cat_exclude": clean_exc
            })

    return unique_segments



def get_optimized_combination_splits(
        df: pd.DataFrame,
        cat_cols: list,
        num_cols: list,
        target_name: str,
        max_segments: int,
        max_depth: int | None
) -> list:
    """
    Trains a DecisionTreeRegressor and parses leaves into a
    representation of numerical bounds and categorical requirements.
    """
    y = df[target_name].values
    X_num = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)

    one_hot_mappings = {}
    X_cat_encoded = pd.DataFrame(index=df.index)

    if cat_cols:
        for col in cat_cols:
            dummies = pd.get_dummies(df[col], prefix=col, prefix_sep="___", dummy_na=False)
            for dummy_col in dummies.columns:
                category_value = dummy_col.split("___")[-1]
                one_hot_mappings[dummy_col] = (col, category_value)
            X_cat_encoded = pd.concat([X_cat_encoded, dummies], axis=1)

    X_total = pd.concat([X_num, X_cat_encoded], axis=1)
    if X_total.empty:
        raise ValueError("No valid overlapping variables detected inside cat_cols or num_cols definitions.")

    clf = DecisionTreeRegressor(
        criterion="squared_error",
        max_leaf_nodes=max_segments,
        max_depth=max_depth,
        min_samples_leaf=1,
        random_state=42
    )
    clf.fit(X_total, y)

    tree_ = clf.tree_
    feature_names = X_total.columns.tolist()

    def get_leaf_paths(node, current_constraints):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            f_name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            if f_name in one_hot_mappings:
                orig_col, cat_val = one_hot_mappings[f_name]
                left_constraints = [dict(c) for c in current_constraints]
                right_constraints = [dict(c) for c in current_constraints]

                left_constraints.append({"type": "cat_exclude", "col": orig_col, "val": cat_val})
                right_constraints.append({"type": "cat_include", "col": orig_col, "val": cat_val})

                return get_leaf_paths(tree_.children_left[node], left_constraints) + \
                    get_leaf_paths(tree_.children_right[node], right_constraints)
            else:
                thresh_rounded = round(float(threshold), 2)
                left_constraints = [dict(c) for c in current_constraints] + [
                    {"type": "num", "col": f_name, "op": "<=", "val": thresh_rounded}]
                right_constraints = [dict(c) for c in current_constraints] + [
                    {"type": "num", "col": f_name, "op": ">", "val": thresh_rounded}]

                return get_leaf_paths(tree_.children_left[node], left_constraints) + \
                    get_leaf_paths(tree_.children_right[node], right_constraints)
        else:
            # Build a structured row definition
            # Rather than strings, we group criteria into dictionary objects
            segment_struct = {
                "num": [],  # list of tuples: (col, op, val)
                "cat_include": [],  # list of tuples: (col, val)
                "cat_exclude": []  # list of tuples: (col, val)
            }
            for c in current_constraints:
                if c["type"] == "num":
                    segment_struct["num"].append((c["col"], c["op"], c["val"]))
                elif c["type"] == "cat_include":
                    segment_struct["cat_include"].append((c["col"], c["val"]))
                elif c["type"] == "cat_exclude":
                    segment_struct["cat_exclude"].append((c["col"], c["val"]))

            return [segment_struct]

    raw_leaf_structures = get_leaf_paths(0, [])

    # Run structural cleanup to minimize, combine, and squash overlaps
    cleaned_segments = reduce_overlapping_segments(raw_leaf_structures)

    return cleaned_segments


def compile_splits_to_string_interval(seg: dict) -> str:
    # Group numeric bounds by feature to display clean ranges if bounded on both sides
    num_by_feature = defaultdict(lambda: {"min": None, "max": None})
    for col, op, val in seg["num"]:
        if op == ">":
            num_by_feature[col]["min"] = val
        elif op == "<=":
            num_by_feature[col]["max"] = val

    num_parts = []
    for col, bounds in num_by_feature.items():
        l_bound = bounds["min"]
        u_bound = bounds["max"]

        if l_bound is not None and u_bound is not None:
            # Collapses sequential splits into a unified bracket string notation
            num_parts.append(f"{col}_({l_bound},{u_bound}]")
        elif l_bound is not None:
            num_parts.append(f"{col}_>{l_bound}")
        elif u_bound is not None:
            num_parts.append(f"{col}_<={u_bound}")

    # Construct categorical representations
    cat_parts = []
    for col, vals in seg["cat_include"]:
        cats_str = ", ".join(f"'{v}'" for v in vals)
        cat_parts.append(f"{col}_IN_({cats_str})")

    for col, vals in seg["cat_exclude"]:
        cats_str = ", ".join(f"'{v}'" for v in vals)
        cat_parts.append(f"{col}_NOT_IN_({cats_str})")

    final_parts = num_parts + cat_parts
    segment_string = "_AND_".join(final_parts) if final_parts else "All_Data"
    return segment_string

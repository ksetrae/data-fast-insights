import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from collections import defaultdict

def get_optimized_combination_segments(
        df: pd.DataFrame,
        cat_cols: list,
        num_cols: list,
        target_data: pd.Series,
        max_segments: int = 6,
        min_segment_size: float = 0.05
) -> pd.Series:
    y = target_data.copy().values

    X_num = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)

    # Keep track of mapping arrays to invert one-hot column names back to original values
    one_hot_mappings = {}
    X_cat_encoded = pd.DataFrame(index=df.index)

    if cat_cols:
        for col in cat_cols:
            # Generate one-hot columns while keeping NaNs cleanly isolated
            dummies = pd.get_dummies(df[col], prefix=col, prefix_sep="___", dummy_na=False)

            # Map the generated one-hot features back to their source category string names
            for dummy_col in dummies.columns:
                # Splitting by "___" and taking the remainder handles category recovery safely
                category_value = dummy_col.split("___")[-1]
                one_hot_mappings[dummy_col] = (col, category_value)

            X_cat_encoded = pd.concat([X_cat_encoded, dummies], axis=1)

    # Combine numerical variables with our mapped binary one-hot switches
    X_total = pd.concat([X_num, X_cat_encoded], axis=1)

    if X_total.empty:
        raise ValueError("No valid overlapping variables detected inside cat_cols or num_cols definitions.")

    # 2. Train the structural tree classifier to maximize information value (IV)
    # FORCE COMBINATIONS: We restrict max_features and drop min_impurity_decrease
    # to force the tree to find deep multi-variable interaction segments.
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_leaf_nodes=max_segments,
        min_samples_leaf=min_segment_size,
        max_features="sqrt",             # Forces the tree to search across different columns
        min_impurity_decrease=0.0,       # Guarantees it keeps splitting up to max_segments
        random_state=42
    )
    clf.fit(X_total, y)

    tree_ = clf.tree_
    feature_names = X_total.columns.tolist()

    # 3. Recursive leaf path traversal tool with integrated categorical grouping
    def get_leaf_paths(node, current_constraints):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            f_name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            # Scenario A: The splitting attribute is a one-hot categorical variable
            if f_name in one_hot_mappings:
                orig_col, cat_val = one_hot_mappings[f_name]

                left_constraints = [dict(c) for c in current_constraints]
                right_constraints = [dict(c) for c in current_constraints]

                # Left child: encoded feature <= 0.5 (Object does NOT match this category)
                left_constraints.append({"type": "cat_exclude", "col": orig_col, "val": cat_val})
                # Right child: encoded feature > 0.5 (Object DOES match this category)
                right_constraints.append({"type": "cat_include", "col": orig_col, "val": cat_val})

                return get_leaf_paths(tree_.children_left[node], left_constraints) + \
                    get_leaf_paths(tree_.children_right[node], right_constraints)

            # Scenario B: The splitting attribute is standard continuous numeric variables
            else:
                thresh_rounded = round(float(threshold), 2)
                left_constraints = [dict(c) for c in current_constraints] + [
                    {"type": "num", "expr": f"{f_name}_<={thresh_rounded}"}]
                # FIX: Added clear '>' operator indicating upper numerical ranges
                right_constraints = [dict(c) for c in current_constraints] + [
                    {"type": "num", "expr": f"{f_name}_{thresh_rounded}"}]

                return get_leaf_paths(tree_.children_left[node], left_constraints) + \
                    get_leaf_paths(tree_.children_right[node], right_constraints)
        else:
            # 4. Collapse and restructure branch constraints into an intuitive segment string
            num_parts = []
            cat_includes = defaultdict(list)
            cat_excludes = defaultdict(list)

            for c in current_constraints:
                if c["type"] == "num":
                    num_parts.append(c["expr"])
                elif c["type"] == "cat_include":
                    cat_includes[c["col"]].append(c["val"])
                elif c["type"] == "cat_exclude":
                    cat_excludes[c["col"]].append(c["val"])

            # Consolidate categorical strings into clear 'IN' or 'NOT IN' statements
            cat_parts = []
            all_cat_cols = set(list(cat_includes.keys()) + list(cat_excludes.keys()))

            for c_col in all_cat_cols:
                if c_col in cat_includes:
                    cats_str = ", ".join(f"'{v}'" for v in sorted(cat_includes[c_col]))
                    cat_parts.append(f"{c_col}_IN_({cats_str})")
                elif c_col in cat_excludes:
                    cats_str = ", ".join(f"'{v}'" for v in sorted(cat_excludes[c_col]))
                    cat_parts.append(f"{c_col}_NOT_IN_({cats_str})")

            final_parts = num_parts + cat_parts
            segment_string = "_AND_".join(final_parts) if final_parts else "All_Data"
            return [(node, segment_string)]

    # Generate the leaf mapping index table using clean strings
    leaf_mappings = dict(get_leaf_paths(0, []))

    # Calculate row leaf assignments instantly
    row_leaf_indices = clf.apply(X_total)

    combination_segments = pd.Series(row_leaf_indices).map(leaf_mappings)
    combination_segments.index = df.index

    return combination_segments

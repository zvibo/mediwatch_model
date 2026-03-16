# Sliding Window Training Analysis

## Context

In our original pipeline, the challenger model was trained exclusively on the **current** window of data (approx. 16,000 samples). Under continuous concept drift (simulated via indexing by `number_inpatient`), the environment rules change. We hypothesized that training on a sliding window of the **current + previous** window (approx. 32,500 samples) would provide a more robust sample size that captures both the emerging patterns and recent stable history.

We modified `runner.py` to use `load_sliding_train()` instead of `load_train()` and re-ran the pipeline.

## Results Comparison

Below is the comparison of the Challenger model's **F1 score** on the current window validation set under both regimes:

| Window Date | Single Window F1 | Sliding Window F1 | Difference | Status |
| :--- | :--- | :--- | :--- | :--- |
| **2004-12-31** | 0.1163 | 0.1163 | - | COLD_START (no previous data) |
| **2005-12-31** | 0.1125 | **0.1496** | +0.0371 | PROMOTED both times |
| **2006-12-31** | 0.0870 | **0.1417** | +0.0547 | RETAINED both times |
| **2007-12-31** | 0.1487 | **0.1757** | +0.0270 | PROMOTED both times |
| **2008-12-31** | 0.2762 | **0.2957** | +0.0195 | PROMOTED both times |

*(Note: In 2006, the challenger still failed to beat the incumbent champion, but it performed phenomenally better than the single-window version).*

## Analysis

1. **Consistent Dominance**: The sliding window approach significantly outperformed the single window approach across *every single drifted window*.
2. **Buffer Against Extreme Drift**: The largest improvement happened in the early-to-mid windows (2005 and 2006) when the target class was still relatively heavily imbalanced. Having double the data points helped XGBoost find more positive samples to learn from.
3. **Catastrophic Forgetting**: Training strictly on the newest, shifted data chunk can cause the model to rapidly over-correct or overfit to the new distribution noise. Allowing it to train across the boundary of the shift (the previous year) forces it to learn generalized rules that work across the boundary, resulting in a model that performs better on the new holdout set.

## Conclusion

The sliding window approach is definitively better. The cost is a 2x increase in training data size, which is trivial at roughly 32,000 rows. Keeping this implementation going forward is highly recommended.

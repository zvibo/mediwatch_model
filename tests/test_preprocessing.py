"""Unit tests for src/preprocessing.py"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import NUMERIC_COLS
from src.preprocessing import (
    CategoricalStringCaster,
    ICD9Binner,
    MissingValueReplacer,
    _bin_icd9,
    build_preprocessor,
    clean_and_engineer,
)

# ── MissingValueReplacer ─────────────────────────────────────────────────────

class TestMissingValueReplacer:
    def test_question_mark_replaced_with_nan(self):
        df = pd.DataFrame({"a": ["?", "hello", "?"], "b": [1, 2, 3]})
        result = MissingValueReplacer().fit_transform(df)
        assert pd.isna(result["a"].iloc[0])
        assert pd.isna(result["a"].iloc[2])

    def test_non_question_mark_values_unchanged(self):
        df = pd.DataFrame({"a": ["hello", "world", "foo"], "b": [1, 2, 3]})
        result = MissingValueReplacer().fit_transform(df)
        assert result["a"].tolist() == ["hello", "world", "foo"]
        assert result["b"].tolist() == [1, 2, 3]

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"a": ["?", "ok"]})
        original_val = df["a"].iloc[0]
        MissingValueReplacer().transform(df)
        assert df["a"].iloc[0] == original_val  # original unchanged

    def test_mixed_values(self):
        df = pd.DataFrame({"col": ["?", "valid", np.nan, "?"]})
        result = MissingValueReplacer().fit_transform(df)
        assert pd.isna(result["col"].iloc[0])
        assert result["col"].iloc[1] == "valid"
        assert pd.isna(result["col"].iloc[2])
        assert pd.isna(result["col"].iloc[3])


# ── ICD9Binner / _bin_icd9 ───────────────────────────────────────────────────

class TestICD9Binner:
    """Test the _bin_icd9 helper directly for mapping logic."""

    def test_circulatory(self):
        assert _bin_icd9(410) == "circulatory"
        assert _bin_icd9(390) == "circulatory"
        assert _bin_icd9(459) == "circulatory"
        assert _bin_icd9(785) == "circulatory"

    def test_diabetes(self):
        assert _bin_icd9(250.0) == "diabetes"
        assert _bin_icd9(250.9) == "diabetes"

    def test_supplementary(self):
        assert _bin_icd9("V01") == "supplementary"
        assert _bin_icd9("V58.11") == "supplementary"

    def test_external(self):
        assert _bin_icd9("E800") == "external"
        assert _bin_icd9("E999") == "external"

    def test_missing(self):
        assert _bin_icd9(np.nan) == "missing"
        assert _bin_icd9(None) == "missing"

    def test_injury(self):
        assert _bin_icd9(999) == "injury"
        assert _bin_icd9(800) == "injury"
        assert _bin_icd9(850) == "injury"

    def test_respiratory(self):
        assert _bin_icd9(460) == "respiratory"
        assert _bin_icd9(519) == "respiratory"
        assert _bin_icd9(786) == "respiratory"

    def test_digestive(self):
        assert _bin_icd9(520) == "digestive"
        assert _bin_icd9(579) == "digestive"

    def test_other(self):
        # 100 doesn't fall in any defined range
        assert _bin_icd9(100) == "other"

    def test_transformer_applies_to_diag_columns(self):
        df = pd.DataFrame({
            "diag_1": ["410", "V01", "E800"],
            "diag_2": ["250", np.nan, "999"],
            "diag_3": ["100", "460", "579"],
        })
        result = ICD9Binner().fit_transform(df)
        assert result["diag_1"].tolist() == ["circulatory", "supplementary", "external"]
        assert result["diag_2"].tolist() == ["diabetes", "missing", "injury"]
        assert result["diag_3"].tolist() == ["other", "respiratory", "digestive"]

    def test_transformer_ignores_non_diag_columns(self):
        df = pd.DataFrame({"other_col": ["410", "999"]})
        result = ICD9Binner().fit_transform(df)
        # Non-diag columns should not be modified
        assert result["other_col"].tolist() == ["410", "999"]


# ── CategoricalStringCaster ──────────────────────────────────────────────────

class TestCategoricalStringCaster:
    def test_id_columns_become_strings(self):
        df = pd.DataFrame({
            "admission_type_id": [1, 2, 3],
            "discharge_disposition_id": [4, 5, 6],
            "admission_source_id": [7, 8, 9],
        })
        result = CategoricalStringCaster().fit_transform(df)
        assert result["admission_type_id"].dtype == object
        assert result["discharge_disposition_id"].dtype == object
        assert result["admission_source_id"].dtype == object

    def test_values_become_string_representations(self):
        df = pd.DataFrame({
            "admission_type_id": [1],
            "discharge_disposition_id": [2],
            "admission_source_id": [3],
        })
        result = CategoricalStringCaster().fit_transform(df)
        assert result["admission_type_id"].iloc[0] == "1"
        assert result["discharge_disposition_id"].iloc[0] == "2"
        assert result["admission_source_id"].iloc[0] == "3"

    def test_non_id_columns_untouched(self):
        df = pd.DataFrame({
            "admission_type_id": [1],
            "some_other_col": [42],
        })
        result = CategoricalStringCaster().fit_transform(df)
        # some_other_col dtype should remain int
        assert result["some_other_col"].dtype != object or result["some_other_col"].iloc[0] == 42

    def test_missing_id_columns_handled_gracefully(self):
        df = pd.DataFrame({"unrelated": [1, 2, 3]})
        # Should not raise even if ID columns are absent
        result = CategoricalStringCaster().fit_transform(df)
        assert list(result.columns) == ["unrelated"]


# ── build_preprocessor ───────────────────────────────────────────────────────

def _make_minimal_df(n=10):
    """Build a small synthetic DataFrame with all expected columns."""
    rng = np.random.default_rng(42)
    data = {}
    # ID cols
    data["encounter_id"] = range(n)
    data["patient_nbr"] = range(n)
    # Numeric cols
    for col in NUMERIC_COLS:
        data[col] = rng.integers(1, 20, size=n).astype(float)
    # Categorical cols (non-ICD9)
    from src.config import CATEGORICAL_COLS
    for col in CATEGORICAL_COLS:
        data[col] = ["A"] * n
    # ICD9 cols
    data["diag_1"] = ["410"] * n
    data["diag_2"] = ["250"] * n
    data["diag_3"] = ["V01"] * n
    # Target
    data["readmitted"] = ["<30"] * (n // 2) + [">30"] * (n - n // 2)
    data["readmitted_binary"] = [1] * (n // 2) + [0] * (n - n // 2)
    return pd.DataFrame(data)


class TestBuildPreprocessor:
    def test_returns_pipeline(self):
        preprocessor = build_preprocessor()
        assert isinstance(preprocessor, Pipeline)

    def test_pipeline_has_expected_steps(self):
        preprocessor = build_preprocessor()
        step_names = [name for name, _ in preprocessor.steps]
        assert "missing_replacer" in step_names
        assert "icd9_binner" in step_names
        assert "cat_caster" in step_names
        assert "column_transformer" in step_names

    def test_fit_transform_produces_array(self):
        preprocessor = build_preprocessor()
        df = _make_minimal_df(n=20)
        X = df.drop(columns=["readmitted", "readmitted_binary"])
        result = preprocessor.fit_transform(X)
        # Should produce a 2D numpy array
        assert hasattr(result, "shape")
        assert result.ndim == 2
        assert result.shape[0] == 20

    def test_output_column_count_matches_feature_cols(self):
        from src.config import FEATURE_COLS
        preprocessor = build_preprocessor()
        df = _make_minimal_df(n=20)
        X = df.drop(columns=["readmitted", "readmitted_binary"])
        result = preprocessor.fit_transform(X)
        # Number of output columns should equal number of feature columns
        assert result.shape[1] == len(FEATURE_COLS)


# ── clean_and_engineer ───────────────────────────────────────────────────────

class TestCleanAndEngineer:
    def test_creates_readmitted_binary_column(self):
        df = pd.DataFrame({"readmitted": ["<30", ">30", "NO", "<30"]})
        result = clean_and_engineer(df)
        assert "readmitted_binary" in result.columns

    def test_less_than_30_maps_to_1(self):
        df = pd.DataFrame({"readmitted": ["<30", "<30"]})
        result = clean_and_engineer(df)
        assert result["readmitted_binary"].tolist() == [1, 1]

    def test_others_map_to_0(self):
        df = pd.DataFrame({"readmitted": [">30", "NO", "never"]})
        result = clean_and_engineer(df)
        assert result["readmitted_binary"].tolist() == [0, 0, 0]

    def test_mixed_values(self):
        df = pd.DataFrame({"readmitted": ["<30", ">30", "NO", "<30"]})
        result = clean_and_engineer(df)
        assert result["readmitted_binary"].tolist() == [1, 0, 0, 1]

    def test_no_readmitted_column_no_crash(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = clean_and_engineer(df)
        assert "readmitted_binary" not in result.columns

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"readmitted": ["<30", ">30"]})
        clean_and_engineer(df)
        assert "readmitted_binary" not in df.columns

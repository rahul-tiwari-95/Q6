"""Portable verification of the one archived float32-mean diagnostic."""
import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def auditor():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        yield importlib.import_module("audit_familiar_starts")


@pytest.fixture(scope="module")
def archived_gaps():
    root = Path(__file__).resolve().parents[1]
    archive = root / "experiments/familiar_starts/pilot_v1"
    with np.load(archive / "fit_predictions.npz") as saved, np.load(archive / "recorded_transitions.npz") as tables:
        rows = saved["bank1_state_rows"]
        predictions = saved["bank1_constrained_bootstrap_seed0"]
        mask = tables["bank1_observed"][rows]
        return predictions.max(1) - np.where(mask, predictions, -np.inf).max(1)


def test_mac_and_linux_float32_means_match_the_same_archived_values(auditor, archived_gaps):
    # Actual saved Mac scalar and Linux CI scalar that exposed the issue.
    for actual in (0.0917019471526146, 0.09170196205377579):
        result = auditor.assert_float32_mean(actual, archived_gaps)
        assert result["canonical_mean"] == pytest.approx(0.09170195541394496, abs=1e-16)
        assert result["absolute_error"] <= result["absolute_tolerance"]
        assert result["error_float32_ulps"] < 2


def test_mean_check_rejects_a_difference_outside_its_scoped_budget(auditor, archived_gaps):
    canonical = float(archived_gaps.mean(dtype=np.float64))
    spacing = float(np.spacing(np.float32(canonical)))
    with pytest.raises(AssertionError):
        auditor.assert_float32_mean(canonical + 4 * spacing, archived_gaps)
    with pytest.raises(AssertionError):
        auditor.assert_float32_mean(canonical + 1e-4, archived_gaps)


def test_zero_gaps_require_an_exact_zero_mean(auditor):
    values = np.zeros(10, np.float32)
    assert auditor.assert_float32_mean(0., values)["absolute_error"] == 0
    with pytest.raises(AssertionError):
        auditor.assert_float32_mean(float(np.nextafter(np.float32(0), np.float32(1))), values)


@pytest.mark.parametrize("values", [np.array([], np.float32), np.array([np.nan], np.float32),
                                    np.array([np.inf], np.float32), np.array([-1.], np.float32),
                                    np.array([1.], np.float64)])
def test_mean_check_rejects_invalid_or_wrong_precision_values(auditor, values):
    with pytest.raises(AssertionError):
        auditor.assert_float32_mean(0., values)


@pytest.mark.parametrize("actual", [float("nan"), float("inf"), -1.])
def test_mean_check_rejects_invalid_reported_scalars(auditor, actual):
    with pytest.raises(AssertionError):
        auditor.assert_float32_mean(actual, np.array([1.], np.float32))

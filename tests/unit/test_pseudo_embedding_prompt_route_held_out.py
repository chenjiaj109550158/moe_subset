import pytest

from pseudoroute.benchmark.pseudo_embedding_prompt_route_held_out import paired_bootstrap


def test_paired_bootstrap_is_deterministic_and_sample_paired() -> None:
    candidate = {
        "a": {"mean_route_hit": 0.7, "mean_selected_mass": 0.8},
        "b": {"mean_route_hit": 0.6, "mean_selected_mass": 0.7},
    }
    previous = {
        "a": {"mean_route_hit": 0.6, "mean_selected_mass": 0.7},
        "b": {"mean_route_hit": 0.5, "mean_selected_mass": 0.6},
    }
    first = paired_bootstrap(candidate, previous, draws=100, seed=17)
    second = paired_bootstrap(candidate, previous, draws=100, seed=17)
    assert first == second
    paired = first["candidate_minus_previous"]
    assert paired["route_hit"]["paired_sample_mean"] == pytest.approx(0.1)
    assert paired["selected_mass"]["paired_sample_mean"] == pytest.approx(0.1)

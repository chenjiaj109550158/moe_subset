from pseudoroute.benchmark.pseudo_embedding_prompt_route_development import (
    realized_window_end,
)


def test_realized_window_end_preserves_partial_final_window() -> None:
    assert realized_window_end(0, 122) == 8
    assert realized_window_end(112, 122) == 120
    assert realized_window_end(120, 122) == 122

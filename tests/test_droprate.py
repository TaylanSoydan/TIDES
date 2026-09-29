"""EigenWorms drop-rate experiment: masks and model sizes (no data download)."""
import droprate


def test_drop_mask():
    L = 17984
    for r in droprate.R_TEST_LIST:
        keep = droprate.drop_mask(L, r, droprate.EVAL_SEEDS[r])
        assert keep == sorted(set(keep)) and keep[0] == 0
        assert abs(len(keep) - round((1 - r) * L)) <= 1
        assert keep == droprate.drop_mask(L, r, droprate.EVAL_SEEDS[r])   # deterministic


def test_ssm_parameter_counts_match_paper():
    expected = {"S5": 29409, "TIDES_Lambda": 29525, "Mamba_S": 29605, "TIDES_BC": 29317,
                "TIDES": 29605, "TIDES_full": 29893}
    for name, n in expected.items():
        model = droprate.build_model(name, "cpu")
        assert sum(p.numel() for p in model.parameters()) == n, name


def test_mamba_parameter_counts():
    expected = {"Mamba": 27125, "Mamba2": 28717, "Mamba3": 26085}
    for name, n in expected.items():
        model = droprate.build_model(name, "cpu", backend="port")
        assert sum(p.numel() for p in model.parameters()) == n, name

"""TIDES model package: step modes, step_scale forms, Hub round trip."""
import numpy as np
import pytest
import torch

from tides import TIDESClassifier, step_scale_from_indices

SMALL = dict(d_input=3, num_classes=4, d_hidden=8, ssm_size=8, ssm_blocks=2, num_blocks=1,
             bidir=True, bc_rank=4)


def _seed(s=0):
    torch.manual_seed(s)
    np.random.seed(s)


@pytest.mark.parametrize("step_mode", ["lti", "input_dependent"])
def test_step_scale_forms(step_mode):
    _seed()
    m = TIDESClassifier(step_mode=step_mode, **SMALL).eval()
    if step_mode == "input_dependent":      # zero-initialised: Δ only matters once learned
        with torch.no_grad():
            m.backbone.blocks[0].ssm.step_proj.weight.normal_()
    x = torch.randn(2, 11, 3)
    keep = [0, 1, 3, 4, 8, 9, 12, 13, 14, 17, 19]
    per_step = step_scale_from_indices(keep)
    with torch.no_grad():
        a = m(x, step_scale=per_step)                        # (L,)
        b = m(x, step_scale=per_step.expand(2, -1))          # (B, L)
        c = m(x)                                             # uniform float 1.0
    assert a.shape == (2, 4) and torch.isfinite(a).all()
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)                          # the gaps matter


def test_step_mode_default_is_unchanged():
    """step_mode='lti' adds no parameters and consumes no extra randomness."""
    _seed(); default = TIDESClassifier(**SMALL)
    _seed(); lti = TIDESClassifier(step_mode="lti", **SMALL)
    assert all(torch.equal(p, q) for p, q in zip(default.state_dict().values(),
                                                  lti.state_dict().values()))
    assert not any("step_proj" in k for k in default.state_dict())


def test_input_dependent_step_starts_at_unit_step():
    _seed()
    m = TIDESClassifier(step_mode="input_dependent", **SMALL)
    proj = m.backbone.blocks[0].ssm.step_proj
    assert torch.all(proj.weight == 0)
    torch.testing.assert_close(torch.nn.functional.softplus(proj.bias),
                               torch.ones_like(proj.bias), atol=1e-4, rtol=0)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        TIDESClassifier(step_mode="learned", **SMALL)


def test_hub_round_trip(tmp_path):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("safetensors")
    _seed()
    m = TIDESClassifier(step_mode="input_dependent", **SMALL).eval()
    m.save_pretrained(tmp_path)
    loaded = TIDESClassifier.from_pretrained(tmp_path).eval()
    x = torch.randn(2, 9, 3)
    with torch.no_grad():
        torch.testing.assert_close(m(x), loaded(x))
    assert loaded.backbone.blocks[0].ssm.step_mode == "input_dependent"


def test_forecasting_hub_round_trip(tmp_path):
    pytest.importorskip("huggingface_hub")
    from tides import TIDESForecastingModel
    _seed()
    m = TIDESForecastingModel(d_input=3, d_hidden=8, ssm_size=8, ssm_blocks=2, num_blocks=1,
                              conj_sym=False, proj_norm="rmsnorm").eval()
    m.save_pretrained(tmp_path)
    loaded = TIDESForecastingModel.from_pretrained(tmp_path).eval()
    values, steps = torch.randn(2, 12, 3), torch.rand(2, 12)
    with torch.no_grad():
        torch.testing.assert_close(m(values, steps), loaded(values, steps))

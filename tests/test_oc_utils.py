from omegaconf import OmegaConf

from reasoning_lab.utils.oc_utils import ensure_native_lora_cfg


def test_ensure_native_lora_cfg_converts_listconfig():
    cfg = OmegaConf.create({
        "r": 8,
        "alpha": 16,
        "dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
    })

    out = ensure_native_lora_cfg(cfg)
    assert isinstance(out, dict)
    assert out["r"] == 8
    assert out["alpha"] == 16
    assert isinstance(out["dropout"], float)
    assert isinstance(out["target_modules"], list)
    assert out["target_modules"] == ["q_proj", "k_proj", "v_proj"]


def test_ensure_native_lora_cfg_handles_missing_fields():
    cfg = OmegaConf.create({})
    out = ensure_native_lora_cfg(cfg)
    assert out.get("target_modules") == ["q_proj", "k_proj", "v_proj", "o_proj"]

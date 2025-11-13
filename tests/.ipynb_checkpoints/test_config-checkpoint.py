from pathlib import Path

from reasoning_lab.config import load_config


def test_load_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
        experiment:
          output_dir: /tmp/out
        """
    )
    cfg = load_config(config_path)
    assert str(cfg.output_dir) == "/tmp/out"

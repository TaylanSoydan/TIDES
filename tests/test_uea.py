"""UEA entry point: seed selection and download cleanup (no data download)."""
import os
import sys

import pytest

import main
import utils

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "uea", "configs", "tides", "SCP1.yaml")


def test_seeds_default_and_explicit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", CONFIG])
    assert main.parse_args().seeds is None                     # -> 42 .. 42 + n_seeds - 1
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", CONFIG, "--seeds", "0", "44"])
    assert main.parse_args().seeds == [0, 44]


def _failing_download(name, extract_path):
    folder = os.path.join(extract_path, name)
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, f"{name}_TRAIN.ts"), "w").close()   # partial download
    raise ValueError("Could not download files")


def test_failed_download_leaves_no_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "load_classification", _failing_download)
    with pytest.raises(ValueError):
        utils.load_uea("Epilepsy", str(tmp_path))
    assert not (tmp_path / "Epilepsy").exists()


def test_failed_load_keeps_existing_folder(tmp_path, monkeypatch):
    (tmp_path / "Epilepsy").mkdir()
    monkeypatch.setattr(utils, "load_classification", _failing_download)
    with pytest.raises(ValueError):
        utils.load_uea("Epilepsy", str(tmp_path))
    assert (tmp_path / "Epilepsy").exists()

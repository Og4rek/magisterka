from __future__ import annotations

import json

import torch

from pcam_repro.cli import (
    _finish_runtime_audit,
    _parser,
    _resume_start_epoch,
    _write_runtime_audit,
)


def test_train_compile_flag_is_explicit_opt_in() -> None:
    parser = _parser()
    eager = parser.parse_args(["train", "--config", "config.toml"])
    compiled = parser.parse_args(
        ["train", "--config", "config.toml", "--compile"]
    )

    assert eager.compile_model is False
    assert compiled.compile_model is True


def test_runtime_audit_retains_eager_to_compiled_transition(tmp_path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    legacy_runtime = {
        "device": "cuda",
        "fast_mode": True,
        "compile_enabled": False,
    }
    (run_directory / "runtime.json").write_text(
        json.dumps(legacy_runtime),
        encoding="utf-8",
    )
    checkpoint = run_directory / "last.pt"

    session_started_at = _write_runtime_audit(
        run_directory,
        {
            "device": "cuda",
            "fast_mode": True,
            "compile_enabled": True,
            "compile_mode": "default",
        },
        checkpoint,
        session_start_epoch=78,
    )

    latest = json.loads(
        (run_directory / "runtime.json").read_text(encoding="utf-8")
    )
    sessions = json.loads(
        (run_directory / "runtime_sessions.json").read_text(encoding="utf-8")
    )
    assert latest["session_start_epoch"] == 78
    assert latest["compile_start_epoch"] == 78
    assert latest["resume_checkpoint"] == str(checkpoint.resolve())
    assert sessions[0] == {"legacy_import": True, **legacy_runtime}
    assert sessions[1]["compile_enabled"] is True
    assert sessions[1]["session_status"] == "started"

    _finish_runtime_audit(run_directory, session_started_at, "completed")
    completed = json.loads(
        (run_directory / "runtime_sessions.json").read_text(encoding="utf-8")
    )[-1]
    assert completed["session_status"] == "completed"
    assert completed["session_finished_at"] is not None


def test_resume_epoch_comes_from_checkpoint_not_history(tmp_path) -> None:
    checkpoint = tmp_path / "last.pt"
    torch.save({"epoch": 77}, checkpoint)
    (tmp_path / "history.json").write_text(
        json.dumps([{"epoch": 78}]),
        encoding="utf-8",
    )

    assert _resume_start_epoch(checkpoint) == 78

import pytest

from transcriber.core.diarize import DiarizationError, _load_error
from transcriber.core.engines import EngineUnavailable


def test_missing_library_is_not_blamed_on_the_token():
    err = _load_error(ModuleNotFoundError("No module named 'matplotlib'"))
    assert isinstance(err, EngineUnavailable)
    assert "matplotlib" in str(err) and "token" not in str(err).lower()


def test_access_errors_point_to_token_and_terms():
    for msg in ("401 Client Error: Unauthorized", "Cannot access gated repo", "403 Forbidden"):
        err = _load_error(OSError(msg))
        assert isinstance(err, DiarizationError) and "Hugging Face" in str(err) and "segmentation-3.0" in str(err)


def test_other_errors_show_the_real_cause_without_token_advice():
    err = _load_error(RuntimeError("CUDA out of memory"))
    assert isinstance(err, DiarizationError)
    assert "out of memory" in str(err) and "Hugging Face" not in str(err)


def test_hf_gated_exception_class_is_recognised():
    hub = pytest.importorskip("huggingface_hub.utils")
    err = _load_error(hub.GatedRepoError("denied"))
    assert "Hugging Face" in str(err)


def test_pyannote_checkpoint_classes_are_allowlisted_for_torch_load():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyannote.audio")
    from transcriber.core.diarize import _allow_pyannote_checkpoints

    _allow_pyannote_checkpoints(torch)
    allowed = {f"{c.__module__}.{c.__name__}" for c in torch.serialization.get_safe_globals()}
    assert {"torch.torch_version.TorchVersion", "pyannote.audio.core.task.Specifications",
            "pyannote.audio.core.task.Problem", "pyannote.audio.core.task.Resolution"} <= allowed


def test_quiet_third_party_warnings_only_hides_ml_library_noise():
    import warnings

    from transcriber.core.device import quiet_third_party_warnings

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        quiet_third_party_warnings()
        # emulate a library warning (module name decides) and one from our own code
        warnings.warn_explicit("old api", UserWarning, "x.py", 1, module="pyannote.audio.core.io")
        warnings.warn_explicit("ours", UserWarning, "y.py", 1, module="transcriber.core.pipeline")
    assert [str(w.message) for w in seen] == ["ours"]

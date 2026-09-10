from unittest.mock import Mock
from types import SimpleNamespace

import pytest

import app

# Test that the predict function uses the local model when use_local is True.
def test_predict_uses_local_model(monkeypatch):
    lyrics = "Driving down a dusty road toward home"
    expected = "Country: 80%, Folk: 15%, Rock: 5%"
    local_model = Mock(return_value=expected)
    remote_model = Mock()

    monkeypatch.setattr(app, "local_generate", local_model)
    monkeypatch.setattr(app, "remote_generate", remote_model)

    result = app.predict(
        lyrics=lyrics,
        temperature=0.7,
        use_local=True,
        hf_token=None,
    )

    assert result == f"Handled by local model: {app.LOCAL_MODEL}\n\n{expected}"
    local_model.assert_called_once()

    prompt, temperature = local_model.call_args.args
    assert lyrics in prompt
    assert "top 3" in prompt
    assert temperature == 0.7
    remote_model.assert_not_called()

# Helper backend function to simulate failures and test fallback behavior.
@pytest.fixture
def backends(monkeypatch):
    local = Mock(return_value="Local prediction")
    remote = Mock(return_value="Remote prediction")
    monkeypatch.setattr(app, "local_generate", local)
    monkeypatch.setattr(app, "remote_generate", remote)
    return local, remote, SimpleNamespace(token="test-token")

# Test that the predict function falls back to the remote model when the local model fails.
@pytest.mark.parametrize("failure, reason", [
    (TimeoutError("private details"), "request timed out"),
    (RuntimeError("private details"), "inference failed"),
])
def test_remote_failure_falls_back_to_local(backends, failure, reason):
    local, remote, token = backends
    remote.side_effect = failure

    result = app.predict("Sample lyrics", 0.7, False, token)

    assert f"Handled by local model: {app.LOCAL_MODEL}" in result
    assert "Automatic fallback" in result
    assert reason in result
    assert result.endswith("Local prediction")
    assert "private details" not in result
    local.assert_called_once_with(*remote.call_args.args[:2])
    remote.assert_called_once()

# Test that the predict function falls back to the remote model when the local model fails.
def test_local_failure_falls_back_to_remote(backends):
    local, remote, token = backends
    local.side_effect = RuntimeError("GPU unavailable")

    result = app.predict("Sample lyrics", 0.7, True, token)

    assert f"Handled by remote model: {app.REMOTE_MODEL}" in result
    assert "Automatic fallback" in result
    assert result.endswith("Remote prediction")
    remote.assert_called_once_with(*local.call_args.args, token)
    local.assert_called_once()

# Test that the predict function returns an error message when both models fail.
def test_both_models_fail_gracefully(backends):
    local, remote, token = backends
    local.side_effect = RuntimeError("private local details")
    remote.side_effect = TimeoutError("private remote details")

    result = app.predict("Sample lyrics", 0.7, False, token)

    assert result.startswith("Unable to generate a prediction")
    assert "Remote model: request timed out" in result
    assert "Local model: inference failed" in result
    assert "private" not in result
    local.assert_called_once()
    remote.assert_called_once()

from unittest.mock import Mock

import app

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

    assert result == expected
    local_model.assert_called_once()

    prompt, temperature = local_model.call_args.args
    assert lyrics in prompt
    assert "top 3" in prompt
    assert temperature == 0.7
    remote_model.assert_not_called()
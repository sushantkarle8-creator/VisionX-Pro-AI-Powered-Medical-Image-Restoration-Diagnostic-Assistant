from io import BytesIO
import json
import urllib.error
import pytest
from visionx.gemini import review_quality
from visionx.images import demo_image

def test_no_consent_no_network(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("Unexpected network"))
    with pytest.raises(ValueError, match="consent"):
        review_quality(demo_image(32), demo_image(32), "test-key", "test-model")

def test_request_and_response(monkeypatch):
    captured = {}
    def fake(request, timeout):
        captured["request"] = request
        return BytesIO(json.dumps({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "Visible smoothing."}]}}]}).encode())
    monkeypatch.setattr("urllib.request.urlopen", fake)
    result = review_quality(demo_image(32), demo_image(32), "test-key", "test-model", consent=True)
    assert result == "Visible smoothing."
    request = captured["request"]
    assert "test-key" not in request.full_url
    assert request.get_header("X-goog-api-key") == "test-key"
    assert len(json.loads(request.data)["contents"][0]["parts"]) == 3

@pytest.mark.parametrize("reason", ["MAX_TOKENS", "SAFETY"])
def test_partial_response_rejected(monkeypatch, reason):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: BytesIO(json.dumps({
        "candidates": [{"finishReason": reason, "content": {"parts": [{"text": "Partial"}]}}]
    }).encode()))
    with pytest.raises(RuntimeError, match="complete"):
        review_quality(demo_image(32), demo_image(32), "test-key", "test-model", consent=True)

def test_http_failure_does_not_expose_key(monkeypatch):
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.com/test-key", 429, "test-key", {}, None)
    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(RuntimeError) as error:
        review_quality(demo_image(32), demo_image(32), "test-key", "test-model", consent=True)
    assert "429" in str(error.value) and "test-key" not in str(error.value)

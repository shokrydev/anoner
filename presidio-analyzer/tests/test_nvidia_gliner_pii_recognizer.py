"""Tests for NvidiaGLiNERPIIRecognizer."""

import sys

import pytest
from unittest.mock import MagicMock, patch

from presidio_analyzer.predefined_recognizers.ner import NvidiaGLiNERPIIRecognizer
from presidio_analyzer.predefined_recognizers.ner.nvidia_gliner_pii_recognizer import (
    NVIDIA_PII_ENTITY_MAPPING,
)


@pytest.fixture
def mock_gliner():
    """Fixture to mock GLiNER class and its methods."""
    pytest.importorskip("gliner", reason="GLiNER package is not installed")

    mock_gliner_instance = MagicMock()
    mock_gliner_instance.to.return_value = mock_gliner_instance

    with patch("gliner.GLiNER.from_pretrained", return_value=mock_gliner_instance):
        yield mock_gliner_instance


def test_initialization_defaults():
    """Test recognizer initializes with correct defaults."""
    recognizer = NvidiaGLiNERPIIRecognizer()

    assert recognizer.model_name == "nvidia/gliner-pii"
    assert recognizer.name == "NvidiaGLiNERPIIRecognizer"
    assert recognizer.threshold == 0.4
    assert recognizer.flat_ner is True


def test_entity_mapping_defaults():
    """Test that NVIDIA PII entity mapping is used by default."""
    recognizer = NvidiaGLiNERPIIRecognizer()

    assert recognizer.model_to_presidio_entity_mapping == NVIDIA_PII_ENTITY_MAPPING
    assert "first_name" in recognizer.gliner_labels
    assert "email" in recognizer.gliner_labels


def test_supported_entities_from_mapping():
    """Test supported entities derived from entity mapping."""
    recognizer = NvidiaGLiNERPIIRecognizer()

    assert "PERSON" in recognizer.supported_entities
    assert "LOCATION" in recognizer.supported_entities
    assert "PHONE_NUMBER" in recognizer.supported_entities
    assert "EMAIL_ADDRESS" in recognizer.supported_entities
    assert "CREDIT_CARD" in recognizer.supported_entities


def test_custom_entity_mapping():
    """Test custom entity mapping overrides default."""
    custom_mapping = {"email": "EMAIL", "first_name": "NAME"}
    recognizer = NvidiaGLiNERPIIRecognizer(entity_mapping=custom_mapping)

    assert recognizer.model_to_presidio_entity_mapping == custom_mapping


def test_custom_threshold():
    """Test custom threshold configuration."""
    recognizer = NvidiaGLiNERPIIRecognizer(threshold=0.7)

    assert recognizer.threshold == 0.7


def test_custom_language():
    """Test language configuration."""
    recognizer = NvidiaGLiNERPIIRecognizer(supported_language="de")

    assert recognizer.supported_language == "de"


def test_inherits_from_gliner_recognizer():
    """Test that NvidiaGLiNERPIIRecognizer inherits from GLiNERRecognizer."""
    from presidio_analyzer.predefined_recognizers.ner import GLiNERRecognizer

    recognizer = NvidiaGLiNERPIIRecognizer()
    assert isinstance(recognizer, GLiNERRecognizer)


def test_analyze_with_mock(mock_gliner):
    """Test analyze method with mocked GLiNER."""
    if sys.version_info < (3, 10):
        pytest.skip("gliner requires Python >= 3.10")

    mock_gliner.predict_entities.return_value = [
        {"label": "first_name", "start": 11, "end": 15, "text": "John", "score": 0.95},
        {"label": "email", "start": 30, "end": 46, "text": "john@example.com", "score": 0.92},
    ]

    recognizer = NvidiaGLiNERPIIRecognizer()
    recognizer.gliner = mock_gliner

    text = "My name is John, email: john@example.com"
    results = recognizer.analyze(text, entities=["PERSON", "EMAIL_ADDRESS"])

    assert len(results) == 2
    assert results[0].entity_type == "PERSON"
    assert results[0].score == pytest.approx(0.95, rel=1e-2)
    assert results[1].entity_type == "EMAIL_ADDRESS"
    assert results[1].score == pytest.approx(0.92, rel=1e-2)


def test_analyze_filters_unrequested_entities(mock_gliner):
    """Test that analyze filters out entities not in the request."""
    if sys.version_info < (3, 10):
        pytest.skip("gliner requires Python >= 3.10")

    mock_gliner.predict_entities.return_value = [
        {"label": "first_name", "start": 11, "end": 15, "text": "John", "score": 0.95},
        {"label": "email", "start": 30, "end": 46, "text": "john@example.com", "score": 0.92},
    ]

    recognizer = NvidiaGLiNERPIIRecognizer()
    recognizer.gliner = mock_gliner

    text = "My name is John, email: john@example.com"
    # Only request PERSON, not EMAIL_ADDRESS
    results = recognizer.analyze(text, entities=["PERSON"])

    assert len(results) == 1
    assert results[0].entity_type == "PERSON"


def test_load_model():
    """Test that model can be loaded (requires model download)."""
    if sys.version_info < (3, 10):
        pytest.skip("gliner requires Python >= 3.10")

    pytest.importorskip("gliner", reason="GLiNER package is not installed")

    recognizer = NvidiaGLiNERPIIRecognizer()
    recognizer.load()

    assert recognizer.gliner is not None


def test_analyze_real_text():
    """Test analyzing real text (requires model download)."""
    if sys.version_info < (3, 10):
        pytest.skip("gliner requires Python >= 3.10")

    pytest.importorskip("gliner", reason="GLiNER package is not installed")

    recognizer = NvidiaGLiNERPIIRecognizer()
    recognizer.load()

    text = "Contact John Smith at john.smith@example.com or call (555) 123-4567"
    results = recognizer.analyze(
        text, entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]
    )

    # Should detect at least some entities
    assert len(results) > 0
    entity_types = [r.entity_type for r in results]
    # At minimum, should find email (most reliable)
    assert any(et in entity_types for et in ["EMAIL_ADDRESS", "PERSON", "PHONE_NUMBER"])

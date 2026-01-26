"""Tests for MinistralOllamaRecognizer."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from presidio_analyzer.predefined_recognizers.ner import MinistralOllamaRecognizer
from presidio_analyzer.predefined_recognizers.ner.ministral_ollama_recognizer import (
    ExtractedEntity,
    OllamaConfig,
    RecognizerConfig,
)


# =============================================================================
# Pydantic Model Tests
# =============================================================================


def test_extracted_entity_valid():
    """Test ExtractedEntity with valid data."""
    entity = ExtractedEntity(text="Max", label="PERSON", start=0, end=3)
    assert entity.text == "Max"
    assert entity.label == "PERSON"
    assert entity.start == 0
    assert entity.end == 3


def test_extracted_entity_invalid_start():
    """Test ExtractedEntity rejects negative start."""
    with pytest.raises(ValidationError):
        ExtractedEntity(text="Max", label="PERSON", start=-1, end=3)


def test_extracted_entity_invalid_end():
    """Test ExtractedEntity rejects end <= start."""
    with pytest.raises(ValidationError):
        ExtractedEntity(text="Max", label="PERSON", start=5, end=3)


def test_ollama_config_defaults():
    """Test OllamaConfig default values."""
    config = OllamaConfig()
    assert "localhost:11434" in str(config.ollama_url)
    assert config.model == "ministral-3:14b"
    assert config.temperature == 0.0
    assert config.timeout == 30.0


def test_ollama_config_custom():
    """Test OllamaConfig with custom values."""
    config = OllamaConfig(
        ollama_url="http://192.168.1.100:11434",
        model="mistral:7b",
        temperature=0.5,
        timeout=60.0,
    )
    assert "192.168.1.100" in str(config.ollama_url)
    assert config.model == "mistral:7b"
    assert config.temperature == 0.5


def test_ollama_config_invalid_temperature():
    """Test OllamaConfig rejects invalid temperature."""
    with pytest.raises(ValidationError):
        OllamaConfig(temperature=3.0)  # Max is 2.0


def test_ollama_config_invalid_timeout():
    """Test OllamaConfig rejects invalid timeout."""
    with pytest.raises(ValidationError):
        OllamaConfig(timeout=-1.0)


def test_recognizer_config_defaults():
    """Test RecognizerConfig default values."""
    config = RecognizerConfig()
    assert "PERSON" in config.supported_entities
    assert config.supported_language == "de"
    assert config.name == "MinistralOllamaRecognizer"


# =============================================================================
# Recognizer Initialization Tests
# =============================================================================


def test_initialization():
    """Test recognizer initializes with default parameters."""
    recognizer = MinistralOllamaRecognizer()

    assert recognizer.ollama_url == "http://localhost:11434"
    assert recognizer.model == "ministral-3:14b"
    assert recognizer.supported_language == "de"
    assert recognizer.temperature == 0.0
    assert recognizer.timeout == 30.0
    assert recognizer.name == "MinistralOllamaRecognizer"
    # Base class calls load() in __init__, so client is initialized
    assert recognizer.client is not None


def test_custom_parameters():
    """Test recognizer accepts custom parameters."""
    recognizer = MinistralOllamaRecognizer(
        ollama_url="http://192.168.1.100:11434",
        model="ministral-3:8b",
        supported_language="en",
        temperature=0.1,
        timeout=60.0,
    )

    assert recognizer.ollama_url == "http://192.168.1.100:11434"
    assert recognizer.model == "ministral-3:8b"
    assert recognizer.supported_language == "en"
    assert recognizer.temperature == 0.1
    assert recognizer.timeout == 60.0


def test_supported_entities():
    """Test default supported entities."""
    recognizer = MinistralOllamaRecognizer()

    assert "PERSON" in recognizer.supported_entities
    assert "LOCATION" in recognizer.supported_entities
    assert "ORGANIZATION" in recognizer.supported_entities
    assert "PHONE_NUMBER" in recognizer.supported_entities
    assert "EMAIL_ADDRESS" in recognizer.supported_entities


def test_custom_supported_entities():
    """Test explicit supported entities."""
    recognizer = MinistralOllamaRecognizer(
        supported_entities=["PERSON", "LOCATION"],
    )

    assert recognizer.supported_entities == ["PERSON", "LOCATION"]


def test_url_trailing_slash_stripped():
    """Test that trailing slash is stripped from URL."""
    recognizer = MinistralOllamaRecognizer(
        ollama_url="http://localhost:11434/",
    )

    assert recognizer.ollama_url == "http://localhost:11434"


def test_invalid_temperature_rejected():
    """Test that invalid temperature is rejected by Pydantic."""
    with pytest.raises(ValidationError):
        MinistralOllamaRecognizer(temperature=5.0)


def test_invalid_timeout_rejected():
    """Test that invalid timeout is rejected by Pydantic."""
    with pytest.raises(ValidationError):
        MinistralOllamaRecognizer(timeout=-10.0)


# =============================================================================
# Prompt Building Tests
# =============================================================================


def test_build_user_prompt():
    """Test user prompt construction."""
    recognizer = MinistralOllamaRecognizer()

    prompt = recognizer._build_user_prompt(
        text="Max Mustermann wohnt in Berlin.",
        entities=["PERSON", "LOCATION"],
    )

    assert "PERSON" in prompt
    assert "LOCATION" in prompt
    assert "Max Mustermann wohnt in Berlin." in prompt


# =============================================================================
# JSON Extraction Tests
# =============================================================================


def test_extract_json_valid_array():
    """Test extracting valid JSON array."""
    recognizer = MinistralOllamaRecognizer()

    response = '[{"text": "Max", "label": "PERSON", "start": 0, "end": 3}]'
    entities = recognizer._extract_json(response)

    assert len(entities) == 1
    assert entities[0]["text"] == "Max"
    assert entities[0]["label"] == "PERSON"


def test_extract_json_with_entities_key():
    """Test extracting JSON with 'entities' key."""
    recognizer = MinistralOllamaRecognizer()

    response = '{"entities": [{"text": "Max", "label": "PERSON", "start": 0, "end": 3}]}'
    entities = recognizer._extract_json(response)

    assert len(entities) == 1
    assert entities[0]["text"] == "Max"


def test_extract_json_from_surrounding_text():
    """Test extracting JSON array from surrounding text."""
    recognizer = MinistralOllamaRecognizer()

    response = 'Here are the entities: [{"text": "Max", "label": "PERSON", "start": 0, "end": 3}] found.'
    entities = recognizer._extract_json(response)

    assert len(entities) == 1
    assert entities[0]["text"] == "Max"


def test_extract_json_invalid():
    """Test handling invalid JSON gracefully."""
    recognizer = MinistralOllamaRecognizer()

    response = "This is not JSON at all"
    entities = recognizer._extract_json(response)

    assert entities == []


def test_extract_json_empty_array():
    """Test extracting empty array."""
    recognizer = MinistralOllamaRecognizer()

    response = "[]"
    entities = recognizer._extract_json(response)

    assert entities == []


# =============================================================================
# Position Validation Tests
# =============================================================================


def test_validate_position_valid():
    """Test position validation with valid entity."""
    recognizer = MinistralOllamaRecognizer()

    text = "Max Mustermann"
    entity = ExtractedEntity(text="Max", label="PERSON", start=0, end=3)

    result = recognizer._validate_position(entity, text)

    assert result is not None
    assert result.text == "Max"
    assert result.start == 0
    assert result.end == 3


def test_validate_position_out_of_bounds_recovers():
    """Test position validation recovers from out-of-bounds by finding actual position."""
    recognizer = MinistralOllamaRecognizer()

    text = "Max"
    entity = ExtractedEntity(text="Max", label="PERSON", start=0, end=100)

    result = recognizer._validate_position(entity, text)

    # Out-of-bounds triggers fallback search, finds "Max" at correct position
    assert result is not None
    assert result.text == "Max"
    assert result.start == 0
    assert result.end == 3


def test_validate_position_out_of_bounds_not_found():
    """Test position validation returns None when text not in document."""
    recognizer = MinistralOllamaRecognizer()

    text = "Hello World"
    entity = ExtractedEntity(text="Max", label="PERSON", start=0, end=100)

    result = recognizer._validate_position(entity, text)

    assert result is None


def test_validate_position_mismatch_corrects():
    """Test position validation corrects position mismatch."""
    recognizer = MinistralOllamaRecognizer()

    text = "Hello Max Mustermann"
    entity = ExtractedEntity(text="Max", label="PERSON", start=0, end=3)

    result = recognizer._validate_position(entity, text)

    assert result is not None
    assert result.start == 6  # Corrected position
    assert result.end == 9


def test_validate_position_text_not_found():
    """Test position validation rejects text not in original."""
    recognizer = MinistralOllamaRecognizer()

    text = "Hello World"
    entity = ExtractedEntity(text="Max", label="PERSON", start=0, end=3)

    result = recognizer._validate_position(entity, text)

    assert result is None


# =============================================================================
# Parse Response Tests (Full Pipeline)
# =============================================================================


def test_parse_response_valid():
    """Test full parse response with valid data."""
    recognizer = MinistralOllamaRecognizer()

    response = '[{"text": "Max", "label": "PERSON", "start": 0, "end": 3}]'
    original_text = "Max Mustermann"

    entities = recognizer._parse_response(response, original_text)

    assert len(entities) == 1
    assert entities[0].text == "Max"
    assert entities[0].label == "PERSON"


def test_parse_response_filters_invalid():
    """Test parse response filters invalid entities."""
    recognizer = MinistralOllamaRecognizer()

    response = json.dumps([
        {"text": "Max", "label": "PERSON", "start": 0, "end": 3},  # Valid
        {"text": "Invalid", "label": "PERSON", "start": 100, "end": 200},  # Out of bounds
        {"text": "Missing", "label": "PERSON"},  # Missing start/end
    ])
    original_text = "Max Mustermann"

    entities = recognizer._parse_response(response, original_text)

    assert len(entities) == 1
    assert entities[0].text == "Max"


# =============================================================================
# Analyze Method Tests
# =============================================================================


@patch("presidio_analyzer.predefined_recognizers.ner.ministral_ollama_recognizer.httpx")
def test_analyze_with_mocked_ollama(mock_httpx):
    """Test analyze method with mocked Ollama response."""
    mock_client = MagicMock()
    mock_httpx.Client.return_value = mock_client

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "response": '[{"text": "Max Mustermann", "label": "PERSON", "start": 0, "end": 14}]'
    }
    mock_client.post.return_value = mock_response

    recognizer = MinistralOllamaRecognizer()

    text = "Max Mustermann wohnt in Berlin."
    results = recognizer.analyze(text, entities=["PERSON", "LOCATION"])

    assert len(results) == 1
    assert results[0].entity_type == "PERSON"
    assert results[0].start == 0
    assert results[0].end == 14
    assert results[0].score == 0.85


@patch("presidio_analyzer.predefined_recognizers.ner.ministral_ollama_recognizer.httpx")
def test_analyze_filters_unrequested_entities(mock_httpx):
    """Test that unrequested entity types are filtered out."""
    mock_client = MagicMock()
    mock_httpx.Client.return_value = mock_client

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "response": json.dumps([
            {"text": "Max", "label": "PERSON", "start": 0, "end": 3},
            {"text": "Berlin", "label": "LOCATION", "start": 13, "end": 19},
        ])
    }
    mock_client.post.return_value = mock_response

    recognizer = MinistralOllamaRecognizer()

    text = "Max wohnt in Berlin."
    results = recognizer.analyze(text, entities=["PERSON"])

    assert all(r.entity_type == "PERSON" for r in results)


@patch("presidio_analyzer.predefined_recognizers.ner.ministral_ollama_recognizer.httpx")
def test_analyze_handles_http_error(mock_httpx):
    """Test graceful handling of HTTP errors."""
    mock_client = MagicMock()
    mock_httpx.Client.return_value = mock_client
    mock_httpx.HTTPError = Exception

    recognizer = MinistralOllamaRecognizer()
    mock_client.post.side_effect = Exception("Connection refused")

    results = recognizer.analyze("Test text", entities=["PERSON"])

    assert results == []


def test_analyze_empty_entities_list():
    """Test analyze returns empty for empty entities list."""
    recognizer = MinistralOllamaRecognizer()

    results = recognizer.analyze("Test text", entities=[])

    assert results == []


def test_analyze_unsupported_entities():
    """Test analyze returns empty for unsupported entities."""
    recognizer = MinistralOllamaRecognizer(
        supported_entities=["PERSON"],
    )

    results = recognizer.analyze("Test text", entities=["UNKNOWN_ENTITY"])

    assert results == []


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
def test_integration_multi_entity_extraction():
    """
    CRITICAL: Verify LLM returns ALL entities, not just one.

    This test catches prompt engineering bugs where the LLM
    only returns a single entity instead of all found entities.

    Run with: poetry run pytest tests/test_ministral_ollama_recognizer.py -m integration -v
    """
    recognizer = MinistralOllamaRecognizer(
        model="ministral-3:8b",
        timeout=60.0,
    )

    text = "Max Mustermann, Tel: +49 89 12345678, wohnt in Berlin."
    results = recognizer.analyze(
        text,
        entities=["PERSON", "PHONE_NUMBER", "LOCATION"],
    )

    assert len(results) >= 2, (
        f"Expected multiple entities, got {len(results)}. "
        "This likely indicates a prompt engineering issue where the LLM "
        "only returns one entity instead of all."
    )

    entity_types = {r.entity_type for r in results}
    assert "PERSON" in entity_types, "Should detect PERSON"


@pytest.mark.integration
def test_integration_analyze_german_clinical_text():
    """Integration test with German clinical text sample."""
    recognizer = MinistralOllamaRecognizer(
        model="ministral-3:8b",
        timeout=60.0,
    )

    text = """Patient: Max Mustermann
Geburtsdatum: 15.03.1965
Anschrift: Hauptstraße 42, 80331 München
Telefon: +49 89 12345678"""

    results = recognizer.analyze(
        text,
        entities=["PERSON", "DATE_TIME", "LOCATION", "PHONE_NUMBER"],
    )

    assert len(results) >= 3, f"Expected at least 3 entities, got {len(results)}"

    entity_types = {r.entity_type for r in results}
    assert "PERSON" in entity_types


@pytest.mark.integration
def test_integration_analyze_english_text():
    """Integration test with English text."""
    recognizer = MinistralOllamaRecognizer(
        model="ministral-3:8b",
        supported_language="en",
        timeout=60.0,
    )

    text = "John Smith works at Microsoft in Seattle."
    results = recognizer.analyze(
        text,
        entities=["PERSON", "ORGANIZATION", "LOCATION"],
    )

    assert len(results) >= 2, f"Expected at least 2 entities, got {len(results)}"

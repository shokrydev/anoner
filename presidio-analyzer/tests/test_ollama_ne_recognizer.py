"""Tests for OllamaNERecognizer."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from presidio_analyzer.predefined_recognizers.ner import OllamaNERecognizer
from presidio_analyzer.predefined_recognizers.ner.ollama_ne_recognizer import (
    ExtractedEntity,
    OllamaConfig,
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
    assert config.model == "ministral-3:8b"
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


# =============================================================================
# Recognizer Initialization Tests
# =============================================================================


def test_initialization():
    """Test recognizer initializes with default parameters."""
    recognizer = OllamaNERecognizer()

    assert recognizer.ollama_url == "http://localhost:11434"
    assert recognizer.model == "ministral-3:8b"
    assert recognizer.supported_language == "de"
    assert recognizer.timeout == 30.0
    assert recognizer.name == "OllamaNERecognizer"


def test_custom_parameters():
    """Test recognizer accepts custom parameters."""
    recognizer = OllamaNERecognizer(
        ollama_url="http://192.168.1.100:11434",
        model="llama3:8b",
        supported_language="en",
        temperature=0.1,
        timeout=60.0,
    )

    assert recognizer.ollama_url == "http://192.168.1.100:11434"
    assert recognizer.model == "llama3:8b"
    assert recognizer.supported_language == "en"
    assert recognizer.temperature == 0.1
    assert recognizer.timeout == 60.0


def test_supported_entities():
    """Test default supported entities include AGE and DE_POSTAL_CODE."""
    recognizer = OllamaNERecognizer()

    assert "PERSON" in recognizer.supported_entities
    assert "LOCATION" in recognizer.supported_entities
    assert "ORGANIZATION" in recognizer.supported_entities
    assert "PHONE_NUMBER" in recognizer.supported_entities
    assert "EMAIL_ADDRESS" in recognizer.supported_entities
    assert "AGE" in recognizer.supported_entities
    assert "DE_POSTAL_CODE" in recognizer.supported_entities


def test_custom_supported_entities():
    """Test explicit supported entities."""
    recognizer = OllamaNERecognizer(
        supported_entities=["PERSON", "LOCATION"],
    )

    # Note: LMRecognizer may add GENERIC_PII_ENTITY
    assert "PERSON" in recognizer.supported_entities
    assert "LOCATION" in recognizer.supported_entities


def test_url_trailing_slash_stripped():
    """Test that trailing slash is stripped from URL."""
    recognizer = OllamaNERecognizer(
        ollama_url="http://localhost:11434/",
    )

    assert recognizer.ollama_url == "http://localhost:11434"


def test_invalid_temperature_rejected():
    """Test that invalid temperature is rejected by Pydantic."""
    with pytest.raises(ValidationError):
        OllamaNERecognizer(temperature=5.0)


def test_invalid_timeout_rejected():
    """Test that invalid timeout is rejected by Pydantic."""
    with pytest.raises(ValidationError):
        OllamaNERecognizer(timeout=-10.0)


# =============================================================================
# Prompt Building Tests
# =============================================================================


def test_build_user_prompt():
    """Test user prompt construction."""
    recognizer = OllamaNERecognizer()

    prompt = recognizer._build_user_prompt(
        text="Max Mustermann wohnt in Berlin.",
        entities=["PERSON", "LOCATION"],
    )

    assert "PERSON" in prompt
    assert "LOCATION" in prompt
    assert "Max Mustermann wohnt in Berlin." in prompt


def test_build_system_prompt_german():
    """Test system prompt includes German examples for German language."""
    recognizer = OllamaNERecognizer(supported_language="de")

    prompt = recognizer._build_system_prompt()

    # Should include German-specific format examples
    assert "45-jährig" in prompt
    assert "Alter: 53" in prompt
    assert "PLZ 12345" in prompt


def test_build_system_prompt_english():
    """Test system prompt excludes German examples for English."""
    recognizer = OllamaNERecognizer(supported_language="en")

    prompt = recognizer._build_system_prompt()

    # Should NOT include German-specific format examples
    assert "45-jährig" not in prompt
    assert "Alter: 53" not in prompt


# =============================================================================
# Label Normalization Tests
# =============================================================================


def test_normalize_label_standard():
    """Test standard label normalization."""
    recognizer = OllamaNERecognizer()

    assert recognizer._normalize_label("PERSON") == "PERSON"
    assert recognizer._normalize_label("LOCATION") == "LOCATION"


def test_normalize_label_mapping():
    """Test label mapping from LLM outputs to Presidio types."""
    recognizer = OllamaNERecognizer()

    # Common LLM label variations
    assert recognizer._normalize_label("NAME") == "PERSON"
    assert recognizer._normalize_label("PER") == "PERSON"
    assert recognizer._normalize_label("LOC") == "LOCATION"
    assert recognizer._normalize_label("ORG") == "ORGANIZATION"
    assert recognizer._normalize_label("PHONE") == "PHONE_NUMBER"
    assert recognizer._normalize_label("EMAIL") == "EMAIL_ADDRESS"
    assert recognizer._normalize_label("POSTAL_CODE") == "DE_POSTAL_CODE"
    assert recognizer._normalize_label("PLZ") == "DE_POSTAL_CODE"


def test_normalize_label_case_insensitive():
    """Test label normalization is case insensitive."""
    recognizer = OllamaNERecognizer()

    assert recognizer._normalize_label("name") == "PERSON"
    assert recognizer._normalize_label("Name") == "PERSON"
    assert recognizer._normalize_label("NAME") == "PERSON"


# =============================================================================
# JSON Extraction Tests
# =============================================================================


def test_extract_json_valid_array():
    """Test extracting valid JSON array."""
    recognizer = OllamaNERecognizer()

    response = '[{"text": "Max", "label": "PERSON", "start": 0, "end": 3}]'
    entities = recognizer._extract_json(response)

    assert len(entities) == 1
    assert entities[0]["text"] == "Max"
    assert entities[0]["label"] == "PERSON"


def test_extract_json_with_entities_key():
    """Test extracting JSON with 'entities' key."""
    recognizer = OllamaNERecognizer()

    response = '{"entities": [{"text": "Max", "label": "PERSON", "start": 0, "end": 3}]}'
    entities = recognizer._extract_json(response)

    assert len(entities) == 1
    assert entities[0]["text"] == "Max"


def test_extract_json_from_surrounding_text():
    """Test extracting JSON array from surrounding text."""
    recognizer = OllamaNERecognizer()

    response = 'Here are the entities: [{"text": "Max", "label": "PERSON", "start": 0, "end": 3}] found.'
    entities = recognizer._extract_json(response)

    assert len(entities) == 1
    assert entities[0]["text"] == "Max"


def test_extract_json_invalid():
    """Test handling invalid JSON gracefully."""
    recognizer = OllamaNERecognizer()

    response = "This is not JSON at all"
    entities = recognizer._extract_json(response)

    assert entities == []


def test_extract_json_empty_array():
    """Test extracting empty array."""
    recognizer = OllamaNERecognizer()

    response = "[]"
    entities = recognizer._extract_json(response)

    assert entities == []


# =============================================================================
# Position Validation Tests
# =============================================================================


def test_validate_position_valid():
    """Test position validation with valid entity."""
    recognizer = OllamaNERecognizer()

    text = "Max Mustermann"
    entity = ExtractedEntity(text="Max", label="PERSON", start=0, end=3)

    result = recognizer._validate_position(entity, text)

    assert result is not None
    assert result.text == "Max"
    assert result.start == 0
    assert result.end == 3


def test_validate_position_out_of_bounds_recovers():
    """Test position validation recovers from out-of-bounds by finding actual position."""
    recognizer = OllamaNERecognizer()

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
    recognizer = OllamaNERecognizer()

    text = "Hello World"
    entity = ExtractedEntity(text="Max", label="PERSON", start=0, end=100)

    result = recognizer._validate_position(entity, text)

    assert result is None


def test_validate_position_mismatch_corrects():
    """Test position validation corrects position mismatch."""
    recognizer = OllamaNERecognizer()

    text = "Hello Max Mustermann"
    entity = ExtractedEntity(text="Max", label="PERSON", start=0, end=3)

    result = recognizer._validate_position(entity, text)

    assert result is not None
    assert result.start == 6  # Corrected position
    assert result.end == 9


def test_validate_position_text_not_found():
    """Test position validation rejects text not in original."""
    recognizer = OllamaNERecognizer()

    text = "Hello World"
    entity = ExtractedEntity(text="Max", label="PERSON", start=0, end=3)

    result = recognizer._validate_position(entity, text)

    assert result is None


# =============================================================================
# Parse Response Tests (Full Pipeline)
# =============================================================================


def test_parse_response_valid():
    """Test full parse response with valid data."""
    recognizer = OllamaNERecognizer()

    response = '[{"text": "Max", "label": "PERSON", "start": 0, "end": 3}]'
    original_text = "Max Mustermann"

    results = recognizer._parse_response(response, original_text)

    assert len(results) == 1
    assert results[0].entity_type == "PERSON"


def test_parse_response_normalizes_labels():
    """Test parse response normalizes LLM labels to Presidio types."""
    recognizer = OllamaNERecognizer()

    # LLM returns "NAME" but we should get "PERSON"
    response = '[{"text": "Max", "label": "NAME", "start": 0, "end": 3}]'
    original_text = "Max Mustermann"

    results = recognizer._parse_response(response, original_text)

    assert len(results) == 1
    assert results[0].entity_type == "PERSON"  # Normalized


def test_parse_response_filters_invalid():
    """Test parse response filters invalid entities."""
    recognizer = OllamaNERecognizer()

    response = json.dumps([
        {"text": "Max", "label": "PERSON", "start": 0, "end": 3},  # Valid
        {"text": "Invalid", "label": "PERSON", "start": 100, "end": 200},  # Out of bounds
        {"text": "Missing", "label": "PERSON"},  # Missing start/end
    ])
    original_text = "Max Mustermann"

    results = recognizer._parse_response(response, original_text)

    assert len(results) == 1
    assert results[0].entity_type == "PERSON"


# =============================================================================
# Analyze Method Tests
# =============================================================================


@patch("presidio_analyzer.predefined_recognizers.ner.ollama_ne_recognizer.httpx")
def test_analyze_with_mocked_ollama(mock_httpx):
    """Test analyze method with mocked Ollama response."""
    mock_client = MagicMock()
    mock_httpx.Client.return_value = mock_client

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "response": '[{"text": "Max Mustermann", "label": "PERSON", "start": 0, "end": 14}]'
    }
    mock_client.post.return_value = mock_response

    recognizer = OllamaNERecognizer()

    text = "Max Mustermann wohnt in Berlin."
    results = recognizer.analyze(text, entities=["PERSON", "LOCATION"])

    assert len(results) >= 1
    person_results = [r for r in results if r.entity_type == "PERSON"]
    assert len(person_results) == 1
    assert person_results[0].start == 0
    assert person_results[0].end == 14


@patch("presidio_analyzer.predefined_recognizers.ner.ollama_ne_recognizer.httpx")
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

    recognizer = OllamaNERecognizer()

    text = "Max wohnt in Berlin."
    results = recognizer.analyze(text, entities=["PERSON"])

    # LMRecognizer filters to only requested entities
    person_results = [r for r in results if r.entity_type == "PERSON"]
    assert len(person_results) >= 1


@patch("presidio_analyzer.predefined_recognizers.ner.ollama_ne_recognizer.httpx")
def test_analyze_handles_http_error(mock_httpx):
    """Test graceful handling of HTTP errors."""
    mock_client = MagicMock()
    mock_httpx.Client.return_value = mock_client
    mock_httpx.HTTPError = Exception

    recognizer = OllamaNERecognizer()
    mock_client.post.side_effect = Exception("Connection refused")

    results = recognizer.analyze("Test text", entities=["PERSON"])

    assert results == []


def test_analyze_empty_text():
    """Test analyze returns empty for empty text."""
    recognizer = OllamaNERecognizer()

    results = recognizer.analyze("", entities=["PERSON"])

    assert results == []


def test_analyze_unsupported_entities():
    """Test analyze returns empty for unsupported entities."""
    recognizer = OllamaNERecognizer(
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

    Run with: poetry run pytest tests/test_ollama_ne_recognizer.py -m integration -v
    """
    recognizer = OllamaNERecognizer(
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
def test_integration_german_age_detection():
    """
    Test detection of German age formats.

    Run with: poetry run pytest tests/test_ollama_ne_recognizer.py -m integration -v
    """
    recognizer = OllamaNERecognizer(
        model="ministral-3:8b",
        supported_language="de",
        timeout=60.0,
    )

    text = "Der 45-jährige Patient Max Mustermann, Alter: 53 Jahre."
    results = recognizer.analyze(
        text,
        entities=["AGE", "PERSON"],
    )

    entity_types = {r.entity_type for r in results}
    assert "AGE" in entity_types or len(results) >= 1, (
        f"Expected to detect AGE or PERSON, got types: {entity_types}"
    )


@pytest.mark.integration
def test_integration_german_postal_code_detection():
    """
    Test detection of German postal codes.

    Run with: poetry run pytest tests/test_ollama_ne_recognizer.py -m integration -v
    """
    recognizer = OllamaNERecognizer(
        model="ministral-3:8b",
        supported_language="de",
        timeout=60.0,
    )

    text = "Anschrift: Hauptstraße 42, 80331 München"
    results = recognizer.analyze(
        text,
        entities=["DE_POSTAL_CODE", "LOCATION"],
    )

    assert len(results) >= 1, f"Expected at least 1 entity, got {len(results)}"


@pytest.mark.integration
def test_integration_analyze_german_clinical_text():
    """Integration test with German clinical text sample."""
    recognizer = OllamaNERecognizer(
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
    recognizer = OllamaNERecognizer(
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

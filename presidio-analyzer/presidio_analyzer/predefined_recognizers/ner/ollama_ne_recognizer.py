"""Ollama-based NER recognizer for local LLM inference.

A general-purpose PII recognizer that works with any Ollama model.
Extends LMRecognizer for shared filtering and entity consolidation.

Requires Ollama to be running:
    systemctl start ollama
    ollama pull ministral-3:8b
"""

import json
import logging
import re
from typing import List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field, HttpUrl, field_validator

from presidio_analyzer import RecognizerResult
from presidio_analyzer.lm_recognizer import LMRecognizer

if TYPE_CHECKING:
    from presidio_analyzer.nlp_engine import NlpArtifacts

try:
    import httpx
except ImportError:
    httpx = None

logger = logging.getLogger("presidio-analyzer")


# Base system prompt for PII extraction
BASE_SYSTEM_PROMPT = """You are a PII extraction system. Extract ALL personally identifiable information from the text.

CRITICAL: Find EVERY occurrence of each entity. If a city name appears 5 times, return 5 separate entries.

Return a JSON array with ALL entities found. Each entity needs:
- text: exact text span as it appears in the original text
- label: entity type (use the exact labels provided)
- start: character start position (0-indexed)
- end: character end position

Return ONLY the JSON array. Be exhaustive - missing PII is worse than false positives."""

# German-specific additions to the prompt
GERMAN_PROMPT_ADDITIONS = """

German-specific patterns to find:
- PERSON: "Herr Bauer", "Frau Klein", "Dr. med. [Name]" - titles with surnames are PERSON
- LOCATION: Streets ("Königsallee 77"), cities ("Frankfurt", "Köln"), hospital wards ("Station 4C")
- ORGANIZATION: Clinics, hospitals, insurance companies, medical practices
- OCCUPATION: Patient jobs in social history section - "Lehrer", "Ingenieurin", "Elektriker"
- ID: Booking numbers, reference numbers, case IDs (format: PREFIX-YEAR-NUMBER or similar)
- AGE: "63-jährig", "44-jährige", "51 Jahre alt", "vor 8 Jahren", "im Alter von 73"
- DATE: "23.09.1990", "am 17.04.2019"

IMPORTANT: Find ALL occurrences of each entity in the text."""


class ExtractedEntity(BaseModel):
    """Pydantic model for an extracted entity from the LLM response."""

    text: str = Field(..., description="The exact text span found")
    label: str = Field(..., description="The entity type label")
    start: int = Field(..., ge=0, description="Character offset where entity starts")
    end: int = Field(..., gt=0, description="Character offset where entity ends")

    @field_validator("end")
    @classmethod
    def end_must_be_greater_than_start(cls, v: int, info) -> int:
        """Validate that end > start."""
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError("end must be greater than start")
        return v


class OllamaConfig(BaseModel):
    """Pydantic configuration for Ollama connection."""

    ollama_url: HttpUrl = Field(
        default="http://localhost:11434",
        description="URL of the Ollama server",
    )
    model: str = Field(
        default="ministral-3:8b",
        description="Ollama model name",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Model temperature (0.0 for deterministic output)",
    )
    timeout: float = Field(
        default=30.0,
        gt=0.0,
        description="HTTP request timeout in seconds",
    )

    @field_validator("ollama_url", mode="before")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        """Remove trailing slash from URL."""
        if isinstance(v, str):
            return v.rstrip("/")
        return v


class OllamaNERecognizer(LMRecognizer):
    """
    Named Entity recognizer using local LLM via Ollama.

    This recognizer connects to a local Ollama server and uses it for
    named entity recognition and PII detection. Works with any Ollama model
    that supports JSON output.

    Requires Ollama to be running with a model pulled:
        ollama pull ministral-3:8b

    Extends LMRecognizer for shared filtering, entity consolidation,
    and result processing.
    """

    # Entity label mapping from common LLM outputs to Presidio types
    LABEL_MAPPING = {
        # Standard mappings
        "NAME": "PERSON",
        "PER": "PERSON",
        "PERSON_NAME": "PERSON",
        "FULL_NAME": "PERSON",
        "LOC": "LOCATION",
        "GPE": "LOCATION",
        "PLACE": "LOCATION",
        "ADDRESS": "LOCATION",
        "ORG": "ORGANIZATION",
        "COMPANY": "ORGANIZATION",
        "PHONE": "PHONE_NUMBER",
        "TELEPHONE": "PHONE_NUMBER",
        "EMAIL": "EMAIL_ADDRESS",
        "MAIL": "EMAIL_ADDRESS",
        "DATE": "DATE_TIME",
        "TIME": "DATE_TIME",
        "DATETIME": "DATE_TIME",
        "BIRTHDAY": "DATE_TIME",
        "DOB": "DATE_TIME",
        "CREDIT_CARD_NUMBER": "CREDIT_CARD",
        "CC": "CREDIT_CARD",
        "BANK_ACCOUNT": "IBAN",
        "POSTAL_CODE": "DE_POSTAL_CODE",
        "PLZ": "DE_POSTAL_CODE",
        "ZIP": "DE_POSTAL_CODE",
        "ZIP_CODE": "DE_POSTAL_CODE",
        "POSTLEITZAHL": "DE_POSTAL_CODE",
    }

    DEFAULT_SUPPORTED_ENTITIES = [
        "PERSON",
        "LOCATION",
        "ORGANIZATION",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "DATE_TIME",
        "CREDIT_CARD",
        "IBAN",
        "AGE",
        "DE_POSTAL_CODE",
    ]

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "ministral-3:8b",
        supported_entities: Optional[List[str]] = None,
        supported_language: str = "de",
        temperature: float = 0.0,
        timeout: float = 30.0,
        min_score: float = 0.4,
        labels_to_ignore: Optional[List[str]] = None,
        name: str = "OllamaNERecognizer",
    ):
        """
        Initialize the Ollama recognizer.

        :param ollama_url: URL of the Ollama server (default: http://localhost:11434)
        :param model: Ollama model name (default: ministral-3:8b)
        :param supported_entities: Presidio entity types to return
        :param supported_language: Language code (default: "de" for German)
        :param temperature: Model temperature (0.0 for deterministic output)
        :param timeout: HTTP request timeout in seconds
        :param min_score: Minimum score threshold for results
        :param labels_to_ignore: Entity labels to skip
        :param name: Recognizer name
        """
        if httpx is None:
            raise ImportError(
                "httpx is required for OllamaNERecognizer. "
                "Install with: pip install httpx"
            )

        # Validate Ollama config with Pydantic
        self.ollama_config = OllamaConfig(
            ollama_url=ollama_url,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )

        self.client: Optional["httpx.Client"] = None
        self._supported_language = supported_language

        super().__init__(
            supported_entities=supported_entities or self.DEFAULT_SUPPORTED_ENTITIES,
            supported_language=supported_language,
            name=name,
            version="1.0.0",
            model_id=model,
            temperature=temperature,
            min_score=min_score,
            labels_to_ignore=labels_to_ignore,
            enable_generic_consolidation=True,
        )

    @property
    def ollama_url(self) -> str:
        """Get the Ollama URL as string."""
        return str(self.ollama_config.ollama_url).rstrip("/")

    @property
    def model(self) -> str:
        """Get the model name."""
        return self.ollama_config.model

    @property
    def timeout(self) -> float:
        """Get the timeout."""
        return self.ollama_config.timeout

    def load(self) -> None:
        """
        Initialize the HTTP client for Ollama.

        Creates a persistent HTTP client for connection pooling.
        """
        self.client = httpx.Client(
            base_url=self.ollama_url,
            timeout=self.timeout,
        )
        logger.info(f"Initialized Ollama client for {self.ollama_url}")

    def _ensure_loaded(self) -> None:
        """Ensure the HTTP client is initialized."""
        if self.client is None:
            self.load()

    def _build_system_prompt(self) -> str:
        """Build the system prompt, adding language-specific examples."""
        prompt = BASE_SYSTEM_PROMPT
        if self._supported_language == "de":
            prompt += GERMAN_PROMPT_ADDITIONS
        return prompt

    def _build_user_prompt(self, text: str, entities: List[str]) -> str:
        """Build the user prompt for entity extraction."""
        entity_list = ", ".join(entities)
        return f"""Extract these PII entity types: {entity_list}

Text to analyze:
{text}

Return JSON array with all entities found:"""

    def _normalize_label(self, label: str) -> str:
        """Normalize LLM label to Presidio entity type."""
        upper_label = label.upper().replace(" ", "_")
        return self.LABEL_MAPPING.get(upper_label, upper_label)

    def _call_llm(
        self,
        text: str,
        entities: List[str],
        **kwargs
    ) -> List[RecognizerResult]:
        """
        Call Ollama API to extract entities.

        Implements the abstract method from LMRecognizer.

        :param text: Text to analyze
        :param entities: Entity types to extract
        :return: List of RecognizerResult objects
        """
        self._ensure_loaded()

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(text, entities)

        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
            "format": "json",
        }

        try:
            response = self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()

            response_text = result.get("response", "")
            results = self._parse_response(response_text, text)
            # Find all occurrences of each detected entity
            results = self._expand_to_all_occurrences(results, text)
            return results

        except httpx.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            return []
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return []

    def _parse_response(
        self, response_text: str, original_text: str
    ) -> List[RecognizerResult]:
        """
        Parse the model response to extract entities.

        :param response_text: Raw response from the model
        :param original_text: Original text for position validation
        :return: List of RecognizerResult objects
        """
        raw_entities = self._extract_json(response_text)
        results = []

        for raw_entity in raw_entities:
            try:
                # Validate with Pydantic
                entity = ExtractedEntity(**raw_entity)

                # Validate/correct position in text
                entity = self._validate_position(entity, original_text)
                if entity is None:
                    continue

                # Normalize the label
                normalized_label = self._normalize_label(entity.label)

                result = RecognizerResult(
                    entity_type=normalized_label,
                    start=entity.start,
                    end=entity.end,
                    score=0.85,  # Default confidence for LLM-based detection
                )
                results.append(result)

            except Exception as e:
                logger.debug(f"Invalid entity skipped: {raw_entity}, error: {e}")
                continue

        return results

    def _expand_to_all_occurrences(
        self, results: List[RecognizerResult], text: str
    ) -> List[RecognizerResult]:
        """
        Find all occurrences of each detected entity text in the original text.

        LLMs often find an entity once but miss repeated occurrences.
        This post-processing step finds all instances of each detected text.

        :param results: Initial results from LLM
        :param text: Original text to search
        :return: Expanded list with all occurrences
        """
        expanded = []
        seen_positions = set()

        # First, add all original results
        for r in results:
            key = (r.start, r.end, r.entity_type)
            if key not in seen_positions:
                seen_positions.add(key)
                expanded.append(r)

        # Then, for each unique entity text, find all occurrences
        unique_entities = {}
        for r in results:
            entity_text = text[r.start:r.end]
            if entity_text and len(entity_text) >= 3:  # Skip very short texts
                if entity_text not in unique_entities:
                    unique_entities[entity_text] = r.entity_type

        for entity_text, entity_type in unique_entities.items():
            start = 0
            while True:
                pos = text.find(entity_text, start)
                if pos == -1:
                    break
                end_pos = pos + len(entity_text)
                key = (pos, end_pos, entity_type)
                if key not in seen_positions:
                    seen_positions.add(key)
                    expanded.append(RecognizerResult(
                        entity_type=entity_type,
                        start=pos,
                        end=end_pos,
                        score=0.80,  # Slightly lower score for expanded matches
                    ))
                start = pos + 1

        return expanded

    def _extract_json(self, response_text: str) -> List[dict]:
        """Extract JSON array from response text."""
        try:
            # Try direct JSON parse first
            data = json.loads(response_text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Handle {"entities": [...]} format
                if "entities" in data:
                    return data["entities"]
                # Handle single entity dict {"text": ..., "label": ...}
                if "text" in data and "label" in data:
                    return [data]
            return []

        except json.JSONDecodeError:
            # Try to extract JSON array from response
            match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Could not parse Ollama response: {response_text[:200]}")
            return []

    def _validate_position(
        self, entity: ExtractedEntity, text: str
    ) -> Optional[ExtractedEntity]:
        """
        Validate and correct entity position in text.

        LLMs often hallucinate positions. This finds the nearest
        occurrence of the entity text to the claimed position.

        :param entity: Entity to validate
        :param text: Original text
        :return: Validated entity or None if text not found
        """
        # Check if text matches claimed position (and bounds are valid)
        if entity.end <= len(text):
            expected_text = text[entity.start : entity.end]
            if entity.text == expected_text:
                return entity

        # Position was wrong/hallucinated - find the NEAREST occurrence
        # to where the LLM claimed it was (most likely the intended match)
        search_start = 0
        best_pos = None
        best_distance = float("inf")

        while True:
            pos = text.find(entity.text, search_start)
            if pos < 0:
                break
            distance = abs(pos - entity.start)
            if distance < best_distance:
                best_distance = distance
                best_pos = pos
            search_start = pos + 1

        if best_pos is not None:
            return ExtractedEntity(
                text=entity.text,
                label=entity.label,
                start=best_pos,
                end=best_pos + len(entity.text),
            )

        return None

    def __del__(self):
        """Clean up HTTP client on destruction."""
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

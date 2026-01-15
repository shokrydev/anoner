"""Ministral-based NER recognizer using Ollama for local LLM inference.

Requires Ollama to be running:
    systemctl start ollama
    ollama pull ministral-3:14b
"""

import json
import logging
import re
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator

from presidio_analyzer import AnalysisExplanation, LocalRecognizer, RecognizerResult

try:
    import httpx
except ImportError:
    httpx = None

logger = logging.getLogger("presidio-analyzer")


# System prompt for PII extraction
DEFAULT_SYSTEM_PROMPT = """You are a PII (Personally Identifiable Information) extraction system.
Your task is to identify PII entities in the given text and return them in JSON format.

For each entity found, return:
- "text": the exact text span found
- "label": one of the requested entity types
- "start": character offset where the entity starts
- "end": character offset where the entity ends

Return ONLY a JSON array of objects. No explanation, no markdown, just the JSON array.
If no entities are found, return an empty array: []

Example output:
[{"text": "John Smith", "label": "PERSON", "start": 0, "end": 10}]"""


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
        default="ministral-3:14b",
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
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt for the model",
    )

    @field_validator("ollama_url", mode="before")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        """Remove trailing slash from URL."""
        if isinstance(v, str):
            return v.rstrip("/")
        return v


class RecognizerConfig(BaseModel):
    """Pydantic configuration for the recognizer."""

    supported_entities: List[str] = Field(
        default=[
            "PERSON",
            "LOCATION",
            "ORGANIZATION",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "DATE_TIME",
            "CREDIT_CARD",
            "IBAN",
            "ID",
        ],
        description="Presidio entity types to detect",
    )
    supported_language: str = Field(
        default="de",
        description="Language code",
    )
    name: str = Field(
        default="MinistralOllamaRecognizer",
        description="Recognizer name",
    )


class MinistralOllamaRecognizer(LocalRecognizer):
    """
    PII recognizer using Ministral model via Ollama.

    This recognizer connects to a local Ollama server running the Ministral model
    and uses it for named entity recognition and PII detection.

    Requires Ollama to be running with the ministral model pulled:
        ollama pull ministral-3:14b
    """

    DEFAULT_SUPPORTED_ENTITIES = [
        "PERSON",
        "LOCATION",
        "ORGANIZATION",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "DATE_TIME",
        "CREDIT_CARD",
        "IBAN",
        "ID",
    ]

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "ministral-3:14b",
        supported_entities: Optional[List[str]] = None,
        supported_language: str = "de",
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 30.0,
        name: str = "MinistralOllamaRecognizer",
    ):
        """
        Initialize the Ministral Ollama recognizer.

        :param ollama_url: URL of the Ollama server (default: http://localhost:11434)
        :param model: Ollama model name (default: ministral-3:14b)
        :param supported_entities: Presidio entity types to return
        :param supported_language: Language code (default: "de" for German)
        :param system_prompt: Custom system prompt for the model
        :param temperature: Model temperature (0.0 for deterministic output)
        :param timeout: HTTP request timeout in seconds
        :param name: Recognizer name
        """
        # Validate Ollama config with Pydantic
        self.ollama_config = OllamaConfig(
            ollama_url=ollama_url,
            model=model,
            temperature=temperature,
            timeout=timeout,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        )

        # Validate recognizer config with Pydantic
        self.recognizer_config = RecognizerConfig(
            supported_entities=supported_entities or self.DEFAULT_SUPPORTED_ENTITIES,
            supported_language=supported_language,
            name=name,
        )

        self.client: Optional["httpx.Client"] = None

        super().__init__(
            supported_entities=self.recognizer_config.supported_entities,
            supported_language=self.recognizer_config.supported_language,
            name=self.recognizer_config.name,
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
    def temperature(self) -> float:
        """Get the temperature."""
        return self.ollama_config.temperature

    @property
    def timeout(self) -> float:
        """Get the timeout."""
        return self.ollama_config.timeout

    @property
    def system_prompt(self) -> str:
        """Get the system prompt."""
        return self.ollama_config.system_prompt

    def load(self) -> None:
        """
        Initialize the HTTP client for Ollama.

        Creates a persistent HTTP client for connection pooling.
        """
        if httpx is None:
            raise ImportError(
                "httpx is not installed. Install with: pip install httpx"
            )

        self.client = httpx.Client(
            base_url=self.ollama_url,
            timeout=self.timeout,
        )
        logger.info(f"Initialized Ollama client for {self.ollama_url}")

    def _ensure_loaded(self) -> None:
        """Ensure the HTTP client is initialized."""
        if self.client is None:
            self.load()

    def _build_user_prompt(self, text: str, entities: List[str]) -> str:
        """Build the user prompt for entity extraction."""
        entity_list = ", ".join(entities)
        return f"""Extract the following PII entity types from the text: {entity_list}

Text to analyze:
{text}

Return only the JSON array:"""

    def _call_ollama(self, text: str, entities: List[str]) -> List[ExtractedEntity]:
        """
        Call Ollama API to extract entities.

        :param text: Text to analyze
        :param entities: Entity types to extract
        :return: List of validated ExtractedEntity objects
        """
        self._ensure_loaded()

        user_prompt = self._build_user_prompt(text, entities)

        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": self.system_prompt,
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
            return self._parse_response(response_text, text)

        except httpx.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            return []
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return []

    def _parse_response(
        self, response_text: str, original_text: str
    ) -> List[ExtractedEntity]:
        """
        Parse the model response to extract and validate entities.

        :param response_text: Raw response from the model
        :param original_text: Original text for position validation
        :return: List of validated ExtractedEntity objects
        """
        raw_entities = self._extract_json(response_text)
        validated_entities = []

        for raw_entity in raw_entities:
            try:
                # Validate with Pydantic
                entity = ExtractedEntity(**raw_entity)

                # Validate position matches text
                entity = self._validate_position(entity, original_text)
                if entity:
                    validated_entities.append(entity)

            except Exception as e:
                logger.debug(f"Invalid entity skipped: {raw_entity}, error: {e}")
                continue

        return validated_entities

    def _extract_json(self, response_text: str) -> List[dict]:
        """Extract JSON array from response text."""
        try:
            # Try direct JSON parse first
            data = json.loads(response_text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "entities" in data:
                return data["entities"]
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

        :param entity: Entity to validate
        :param text: Original text
        :return: Validated entity or None if invalid
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

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts=None,
    ) -> List[RecognizerResult]:
        """
        Analyze text for PII entities using Ministral via Ollama.

        :param text: Text to analyze
        :param entities: Entity types to detect
        :param nlp_artifacts: Not used (standalone model)
        :return: List of RecognizerResult objects
        """
        # Filter to requested entities that we support
        requested_entities = [e for e in entities if e in self.supported_entities]
        if not requested_entities:
            return []

        # Call Ollama
        extracted_entities = self._call_ollama(text, requested_entities)

        results = []
        for entity in extracted_entities:
            if entity.label not in requested_entities:
                continue

            result = RecognizerResult(
                entity_type=entity.label,
                start=entity.start,
                end=entity.end,
                score=0.85,  # Default confidence for LLM-based detection
                analysis_explanation=AnalysisExplanation(
                    recognizer=self.name,
                    original_score=0.85,
                    textual_explanation=f"Identified as {entity.label} by {self.model}",
                ),
            )
            results.append(result)

        return results

    def __del__(self):
        """Clean up HTTP client on destruction."""
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

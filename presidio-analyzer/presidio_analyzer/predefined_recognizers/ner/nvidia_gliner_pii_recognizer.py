"""NVIDIA GLiNER-PII recognizer for PII detection with automatic text chunking.

A wrapper around GLiNERRecognizer that uses nvidia/gliner-pii model
with its 55+ PII entity types. Automatically chunks long texts to handle
the model's 384 token limit.

Note: This recognizer requires the `gliner` library which has dependency warnings:
- FutureWarning from huggingface_hub (`resume_download` deprecated)
- DeprecationWarning for SwigPyPacked/SwigPyObject from onnxruntime

These are upstream issues in gliner's dependencies, not in this code.

See: https://huggingface.co/nvidia/gliner-pii
"""

import logging
from typing import Dict, List, Optional, Tuple

from presidio_analyzer import RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

from presidio_analyzer.predefined_recognizers.ner.gliner_recognizer import (
    GLiNERRecognizer,
)

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

logger = logging.getLogger("presidio-analyzer")

# Max tokens for GLiNER models
MAX_TOKENS = 384

# Mapping from nvidia/gliner-pii labels to Presidio entity types
NVIDIA_PII_ENTITY_MAPPING: Dict[str, str] = {
    # Names
    "first_name": "PERSON",
    "last_name": "PERSON",
    "user_name": "PERSON",
    # Location
    "street_address": "LOCATION",
    "city": "LOCATION",
    "state": "LOCATION",
    "country": "LOCATION",
    "county": "LOCATION",
    "postcode": "LOCATION",
    "coordinate": "LOCATION",
    # Organization
    "company_name": "ORGANIZATION",
    # Contact
    "phone_number": "PHONE_NUMBER",
    "email": "EMAIL_ADDRESS",
    "fax_number": "PHONE_NUMBER",
    "url": "URL",
    # Dates
    "date_of_birth": "DATE_TIME",
    "date": "DATE_TIME",
    "time": "DATE_TIME",
    "date_time": "DATE_TIME",
    "age": "AGE",
    # Financial
    "ssn": "US_SSN",
    "credit_debit_card": "CREDIT_CARD",
    "account_number": "US_BANK_NUMBER",
    "bank_routing_number": "ABA_ROUTING_NUMBER",
    "swift_bic": "IBAN_CODE",
    "cvv": "CREDIT_CARD",
    "pin": "PIN",
    # IDs
    "medical_record_number": "MEDICAL_RECORD",
    "health_plan_beneficiary_number": "HEALTH_PLAN_ID",
    "customer_id": "ID",
    "employee_id": "ID",
    "certificate_license_number": "LICENSE",
    "license_plate": "LICENSE_PLATE",
    "vehicle_identifier": "VEHICLE_ID",
    # Tech
    "ipv4": "IP_ADDRESS",
    "mac_address": "MAC_ADDRESS",
    "device_identifier": "DEVICE_ID",
    "http_cookie": "COOKIE",
    "password": "PASSWORD",
    # Demographics
    "gender": "DEMOGRAPHIC",
    "race_ethnicity": "DEMOGRAPHIC",
    "blood_type": "MEDICAL",
    "sexuality": "DEMOGRAPHIC",
    "religious_belief": "DEMOGRAPHIC",
    "political_view": "DEMOGRAPHIC",
    "political_affiliation": "DEMOGRAPHIC",
    "language": "DEMOGRAPHIC",
    # Other
    "occupation": "OCCUPATION",
    "employment_status": "EMPLOYMENT",
    "education_level": "EDUCATION",
    "biometric_identifier": "BIOMETRIC",
}


class NvidiaGLiNERPIIRecognizer(GLiNERRecognizer):
    """GLiNER recognizer using NVIDIA's gliner-pii model with automatic chunking.

    Pre-trained on 55+ PII/PHI entity types for privacy protection.
    Supports zero-shot NER - can detect entities not seen during training.
    Automatically chunks long texts to handle the 384 token model limit.

    See: https://huggingface.co/nvidia/gliner-pii
    """

    def __init__(
        self,
        supported_entities: Optional[List[str]] = None,
        name: str = "NvidiaGLiNERPIIRecognizer",
        supported_language: str = "en",
        version: str = "0.0.1",
        context: Optional[List[str]] = None,
        entity_mapping: Optional[Dict[str, str]] = None,
        model_name: str = "nvidia/gliner-pii",
        flat_ner: bool = True,
        multi_label: bool = False,
        threshold: float = 0.4,
        map_location: Optional[str] = None,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        """Initialize NVIDIA GLiNER-PII recognizer.

        :param supported_entities: List of Presidio entity types to return.
            If None, uses entities from entity_mapping.
        :param name: Recognizer name
        :param supported_language: Language code (model works on multiple languages)
        :param version: Recognizer version
        :param context: Context words (N/A for this recognizer)
        :param entity_mapping: GLiNER label to Presidio entity mapping.
            If None, uses NVIDIA_PII_ENTITY_MAPPING.
        :param model_name: HuggingFace model name
        :param flat_ner: Use flat NER without nested entities
        :param multi_label: Allow multiple labels per span
        :param threshold: Minimum confidence score (default: 0.4)
        :param map_location: Device for inference (None for auto-detect)
        :param chunk_size: Tokens per chunk for long texts (default: 300)
        :param chunk_overlap: Overlapping tokens between chunks (default: 50)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._tokenizer = None

        super().__init__(
            supported_entities=supported_entities,
            name=name,
            supported_language=supported_language,
            version=version,
            context=context,
            entity_mapping=entity_mapping or NVIDIA_PII_ENTITY_MAPPING,
            model_name=model_name,
            flat_ner=flat_ner,
            multi_label=multi_label,
            threshold=threshold,
            map_location=map_location,
        )

    def load(self) -> None:
        """Load the GLiNER model and tokenizer for chunking."""
        super().load()

        if AutoTokenizer:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except Exception as e:
                logger.warning(
                    f"Could not load tokenizer for {self.model_name}: {e}. "
                    "Using character-based chunking."
                )
                self._tokenizer = None
        else:
            logger.warning("transformers not installed, using character-based chunking")

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ) -> List[RecognizerResult]:
        """Analyze text with automatic chunking for long texts.

        :param text: The text to be analyzed
        :param entities: The list of entities this recognizer is requested to return
        :param nlp_artifacts: N/A for this recognizer
        """
        token_count = self._get_token_count(text)

        if token_count <= MAX_TOKENS:
            return super().analyze(text, entities, nlp_artifacts)

        logger.info(f"Chunking text ({token_count} tokens) into {self.chunk_size}-token pieces")
        return self._analyze_chunked(text, entities)

    # --- Chunking implementation ---

    def _get_token_count(self, text: str) -> int:
        """Get token count using tokenizer or estimate."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return len(text) // 4  # ~4 chars per token estimate

    def _analyze_chunked(self, text: str, entities: List[str]) -> List[RecognizerResult]:
        """Run analysis on chunks and merge results."""
        chunks = self._create_chunks(text)
        all_results = []

        for chunk_text, offset in chunks:
            results = super().analyze(chunk_text, entities, None)
            for r in results:
                all_results.append(
                    RecognizerResult(
                        entity_type=r.entity_type,
                        start=r.start + offset,
                        end=r.end + offset,
                        score=r.score,
                        analysis_explanation=r.analysis_explanation,
                    )
                )

        return self._merge_results(all_results)

    def _create_chunks(self, text: str) -> List[Tuple[str, int]]:
        """Split text into overlapping chunks. Returns (chunk_text, char_offset) pairs."""
        if self._tokenizer:
            encoding = self._tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=False
            )
            offsets = encoding["offset_mapping"]
            chunks = []
            i = 0

            while i < len(offsets):
                end = min(i + self.chunk_size, len(offsets))
                char_start, char_end = offsets[i][0], offsets[end - 1][1]
                chunks.append((text[char_start:char_end], char_start))

                if end >= len(offsets):
                    break
                i = end - self.chunk_overlap

            return chunks

        # Fallback: character-based chunking
        chunk_chars = self.chunk_size * 4
        overlap_chars = self.chunk_overlap * 4
        chunks = []
        i = 0

        while i < len(text):
            end = min(i + chunk_chars, len(text))
            if end < len(text):
                space = text.rfind(" ", i + chunk_chars // 2, end)
                if space > i:
                    end = space
            chunks.append((text[i:end], i))
            if end >= len(text):
                break
            i = end - overlap_chars

        return chunks

    def _merge_results(self, results: List[RecognizerResult]) -> List[RecognizerResult]:
        """Merge overlapping results, keeping highest scores."""
        if not results:
            return []

        results.sort(key=lambda r: (r.start, -r.score))
        merged = []

        for r in results:
            duplicate = False
            for m in merged:
                if r.entity_type == m.entity_type and self._overlaps(r, m):
                    if r.score > m.score:
                        m.start, m.end, m.score = r.start, r.end, r.score
                    duplicate = True
                    break
            if not duplicate:
                merged.append(r)

        return merged

    def _overlaps(self, a: RecognizerResult, b: RecognizerResult) -> bool:
        """Check if two results overlap by >50%."""
        overlap = max(0, min(a.end, b.end) - max(a.start, b.start))
        min_len = min(a.end - a.start, b.end - b.start)
        return overlap > min_len * 0.5

"""NVIDIA GLiNER-PII recognizer for PII detection.

A thin wrapper around GLiNERRecognizer that uses nvidia/gliner-pii model
with its 55+ PII entity types.

Note: This recognizer requires the `gliner` library which has dependency warnings due to
    it inheriting from GLiNERRecognizer:
- FutureWarning from huggingface_hub (`resume_download` deprecated) - gliner
  uses huggingface_hub for model downloads with a deprecated parameter
- DeprecationWarning for SwigPyPacked/SwigPyObject - gliner uses onnxruntime
  which has SWIG-generated C++ bindings that lack proper __module__ attributes

These are upstream issues in gliner's dependencies, not in this code.

See: https://huggingface.co/nvidia/gliner-pii
"""

from typing import Dict, List, Optional

from presidio_analyzer.predefined_recognizers.ner.gliner_recognizer import (
    GLiNERRecognizer,
)

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
    """
    GLiNER recognizer using NVIDIA's gliner-pii model.

    Pre-trained on 55+ PII/PHI entity types for privacy protection.
    Supports zero-shot NER - can detect entities not seen during training.

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
    ):
        """
        Initialize NVIDIA GLiNER-PII recognizer.

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
        """

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

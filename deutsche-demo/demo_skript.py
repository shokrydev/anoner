#!/usr/bin/env python3
"""
Demo: Vergleich der Erkennungsmethoden für deutsche klinische Texte.

Zeigt separat: Pattern-only vs. NER-only vs. LLM-only vs. Regex+LLM

Siehe README.md für Voraussetzungen und Verwendung.
"""

import html
import logging
import sys
import warnings
from pathlib import Path

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

# Add local packages to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "presidio-analyzer"))
sys.path.insert(0, str(PROJECT_ROOT / "presidio-anonymizer"))

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Import recognizers
from presidio_analyzer.predefined_recognizers.ner import (
    OllamaNERecognizer,
    NvidiaGLiNERPIIRecognizer,
)

# Import German PII recognizers
from presidio_analyzer.predefined_recognizers import (
    DeBsnrRecognizer,
    DeKvnrRecognizer,
    DeLanrRecognizer,
    DePersonalIdRecognizer,
    DePostalCodeRecognizer,
    DeSocialSecurityRecognizer,
    DeTaxIdRecognizer,
    DeTelematikIdRecognizer,
)

# Import utils for metrics and report generation
from utils.metrics import AnnotatedEntity, evaluate_results, load_ground_truth, get_missed_entities
from utils.report_template import generate_report

# Default test input file
DEFAULT_INPUT_FILE = Path(__file__).parent / "eingabe" / "entlassungsbrief.txt"

# Entity types to detect
ENTITIES = [
    "PERSON", "LOCATION", "ORGANIZATION", "PHONE_NUMBER", "EMAIL_ADDRESS",
    "DATE_TIME", "AGE", "OCCUPATION", "ID",
    "DE_KVNR", "DE_LANR", "DE_BSNR", "DE_TELEMATIK_ID",
    "DE_PERSONAL_ID", "DE_TAX_ID", "DE_SOCIAL_SECURITY", "DE_POSTAL_CODE",
]

# Entity domains for fair evaluation
PATTERN_DOMAIN = {
    "DE_KVNR", "DE_LANR", "DE_BSNR", "DE_TELEMATIK_ID",
    "DE_PERSONAL_ID", "DE_TAX_ID", "DE_SOCIAL_SECURITY", "DE_POSTAL_CODE",
}
LLM_DOMAIN = {
    "PERSON", "LOCATION", "ORGANIZATION", "PHONE_NUMBER", "EMAIL_ADDRESS",
    "DATE_TIME", "AGE", "OCCUPATION", "ID",
}

# Entity type to color mapping for HTML report
ENTITY_COLORS = {
    "PERSON": "#ff6b6b",
    "LOCATION": "#4dabf7",
    "DE_POSTAL_CODE": "#74c0fc",
    "ORGANIZATION": "#b197fc",
    "DE_KVNR": "#69db7c",
    "DE_LANR": "#8ce99a",
    "DE_BSNR": "#a9e34b",
    "DE_TELEMATIK_ID": "#c0eb75",
    "DE_PERSONAL_ID": "#ffa94d",
    "DE_TAX_ID": "#ffc078",
    "DE_SOCIAL_SECURITY": "#ffec99",
    "PHONE_NUMBER": "#63e6be",
    "EMAIL_ADDRESS": "#38d9a9",
    "DATE_TIME": "#f783ac",
    "AGE": "#fab005",
    "OCCUPATION": "#fd7e14",
    "ID": "#e8590c",
}

ENTITY_LABELS_DE = {
    "PERSON": "Person",
    "LOCATION": "Ort",
    "ORGANIZATION": "Einrichtung",
    "DE_KVNR": "KVNR",
    "DE_LANR": "LANR",
    "DE_BSNR": "BSNR",
    "DE_TELEMATIK_ID": "Telematik-ID",
    "DE_PERSONAL_ID": "Ausweis-Nr",
    "DE_TAX_ID": "Steuer-ID",
    "DE_SOCIAL_SECURITY": "Sozialvers.-Nr",
    "DE_POSTAL_CODE": "PLZ",
    "PHONE_NUMBER": "Telefon",
    "EMAIL_ADDRESS": "E-Mail",
    "DATE_TIME": "Datum",
    "AGE": "Alter",
    "OCCUPATION": "Beruf",
    "ID": "Kennung",
}


def load_input_text(file_path: Path = DEFAULT_INPUT_FILE) -> str:
    """Load input text from file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


def get_nlp_engine():
    """Get SpaCy NLP engine for context enhancement."""
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "de", "model_name": "de_core_news_sm"}],
    }
    try:
        return NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
    except Exception as e:
        print(f"  ✗ SpaCy NLP engine failed: {e}")
        return None


def setup_pattern_analyzer(nlp_engine) -> AnalyzerEngine:
    """Set up analyzer with only German pattern recognizers."""
    registry = RecognizerRegistry(supported_languages=["de"])

    german_recognizers = [
        DeKvnrRecognizer, DeLanrRecognizer, DeBsnrRecognizer,
        DeTelematikIdRecognizer, DePersonalIdRecognizer, DeTaxIdRecognizer,
        DeSocialSecurityRecognizer, DePostalCodeRecognizer,
    ]

    for recognizer_class in german_recognizers:
        try:
            registry.add_recognizer(recognizer_class())
        except Exception:
            pass

    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["de"],
    )


def setup_ner_analyzer(nlp_engine) -> AnalyzerEngine:
    """Set up analyzer with only NvidiaGLiNERPIIRecognizer."""
    registry = RecognizerRegistry(supported_languages=["de"])

    try:
        gliner_recognizer = NvidiaGLiNERPIIRecognizer(supported_language="de")
        registry.add_recognizer(gliner_recognizer)
    except Exception as e:
        print(f"  ✗ NvidiaGLiNERPIIRecognizer failed: {e}")

    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["de"],
    )


def setup_llm_analyzer(nlp_engine, model: str = "ministral-3:8b") -> AnalyzerEngine:
    """Set up analyzer with only OllamaNERecognizer."""
    registry = RecognizerRegistry(supported_languages=["de"])

    # Larger models need more time
    timeout = 300.0 if "14b" in model else 120.0

    try:
        ollama_recognizer = OllamaNERecognizer(
            ollama_url="http://localhost:11434",
            model=model,
            supported_language="de",
            timeout=timeout,
        )
        registry.add_recognizer(ollama_recognizer)
    except Exception as e:
        print(f"  ✗ OllamaNERecognizer failed: {e}")

    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["de"],
    )


def analyze_text(analyzer: AnalyzerEngine, text: str) -> list:
    """Analyze text for PII entities."""
    return analyzer.analyze(text=text, language="de", entities=ENTITIES)


def merge_results(results_list: list[list]) -> list:
    """Merge results from multiple analyzers."""
    all_results = []
    for results in results_list:
        all_results.extend(results)
    return all_results


def remove_overlapping_entities(results: list) -> list:
    """Remove overlapping entities, keeping highest score for each span."""
    if not results:
        return []

    sorted_results = sorted(results, key=lambda r: (r.start, -r.score))
    filtered = []

    for r in sorted_results:
        overlaps = False
        for accepted in filtered:
            if r.start < accepted.end and r.end > accepted.start:
                overlaps = True
                break
        if not overlaps:
            filtered.append(r)

    return sorted(filtered, key=lambda r: r.start)


def get_anonymization_operators() -> dict:
    """Define anonymization operators for PII entity types."""
    return {
        "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
        "LOCATION": OperatorConfig("replace", {"new_value": "[ORT]"}),
        "ORGANIZATION": OperatorConfig("replace", {"new_value": "[EINRICHTUNG]"}),
        "DE_KVNR": OperatorConfig("replace", {"new_value": "[KVNR]"}),
        "DE_LANR": OperatorConfig("replace", {"new_value": "[LANR]"}),
        "DE_BSNR": OperatorConfig("replace", {"new_value": "[BSNR]"}),
        "DE_TELEMATIK_ID": OperatorConfig("replace", {"new_value": "[TELEMATIK-ID]"}),
        "DE_PERSONAL_ID": OperatorConfig("replace", {"new_value": "[AUSWEIS-NR]"}),
        "DE_TAX_ID": OperatorConfig("replace", {"new_value": "[STEUER-ID]"}),
        "DE_SOCIAL_SECURITY": OperatorConfig("replace", {"new_value": "[SVNR]"}),
        "DE_POSTAL_CODE": OperatorConfig("replace", {"new_value": "[PLZ]"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[TELEFON]"}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
        "DATE_TIME": OperatorConfig("replace", {"new_value": "[DATUM]"}),
        "AGE": OperatorConfig("replace", {"new_value": "[ALTER]"}),
        "OCCUPATION": OperatorConfig("replace", {"new_value": "[BERUF]"}),
        "ID": OperatorConfig("replace", {"new_value": "[KENNUNG]"}),
        "DEFAULT": OperatorConfig("replace", {"new_value": "[PII]"}),
    }


def build_highlighted_text(text: str, results: list) -> str:
    """Build HTML highlighted text from results."""
    filtered = remove_overlapping_entities(results)
    parts = []
    last_end = 0

    for r in filtered:
        if r.start > last_end:
            parts.append(html.escape(text[last_end:r.start]))
        elif r.start < last_end:
            continue

        color = ENTITY_COLORS.get(r.entity_type, "#e9ecef")
        entity_text = html.escape(text[r.start:r.end])
        label = ENTITY_LABELS_DE.get(r.entity_type, r.entity_type)
        parts.append(
            f'<mark class="entity" style="background-color: {color};" '
            f'data-entity="{r.entity_type}" title="{label} (Score: {r.score:.2f})">'
            f'{entity_text}</mark>'
        )
        last_end = r.end

    if last_end < len(text):
        parts.append(html.escape(text[last_end:]))

    return "".join(parts)


def build_legend(results: list) -> str:
    """Build HTML legend for results."""
    filtered = remove_overlapping_entities(results)
    entity_counts = {}
    for r in filtered:
        entity_counts[r.entity_type] = entity_counts.get(r.entity_type, 0) + 1

    if not entity_counts:
        return '<span class="no-results">Keine Entitäten erkannt</span>'

    items = []
    for entity_type, count in sorted(entity_counts.items()):
        color = ENTITY_COLORS.get(entity_type, "#e9ecef")
        label = ENTITY_LABELS_DE.get(entity_type, entity_type)
        items.append(
            f'<span class="legend-item">'
            f'<span class="legend-color" style="background-color: {color};"></span>'
            f'{label} ({count})</span>'
        )
    return "\n".join(items)


def results_to_annotated_entities(results: list, text: str) -> list[AnnotatedEntity]:
    """Convert Presidio results to AnnotatedEntity list for metrics calculation."""
    filtered = remove_overlapping_entities(results)
    return [
        AnnotatedEntity(
            start=r.start,
            end=r.end,
            entity_type=r.entity_type,
            text=text[r.start:r.end],
        )
        for r in filtered
    ]


def print_results_summary(
    name: str,
    results: list,
    metrics: dict | None = None,
    domain_metrics: dict | None = None,
) -> None:
    """Print a summary of detected entities with optional metrics."""
    filtered = remove_overlapping_entities(results)
    by_type = {}
    for r in filtered:
        if r.entity_type not in by_type:
            by_type[r.entity_type] = []
        by_type[r.entity_type].append(r)

    print(f"\n  {name}: {len(filtered)} Entitäten")
    for entity_type, entities in sorted(by_type.items()):
        print(f"    - {entity_type}: {len(entities)}")

    if metrics:
        print(f"    → Gesamt: P={metrics['precision']:.1%} R={metrics['recall']:.1%} F2={metrics['f_beta']:.1%}")

    if domain_metrics:
        for domain_name, m in domain_metrics.items():
            if m and m['tp'] + m['fp'] + m['fn'] > 0:
                print(f"    → {domain_name}: P={m['precision']:.1%} R={m['recall']:.1%} F2={m['f_beta']:.1%}")


def main(input_file: Path = DEFAULT_INPUT_FILE, overlap_threshold: float = 0.5, model: str = "ministral-3:8b"):
    """Main function to run the comparison pipeline."""
    print("\n" + "=" * 60)
    print("GERMAN CLINICAL TEXT - METHODEN-VERGLEICH")
    print("=" * 60)

    # Load input text
    print(f"\n[1/7] Lade Text: {input_file.name}")
    text = load_input_text(input_file)
    print(f"      {len(text)} Zeichen, ~{len(text.split())} Wörter")

    # Load ground truth annotations if available
    annotations_file = input_file.with_name(input_file.stem + "_annotations.json")
    ground_truth = None
    if annotations_file.exists():
        print(f"\n[2/7] Lade Ground Truth: {annotations_file.name}")
        ground_truth = load_ground_truth(annotations_file)
        print(f"      {len(ground_truth)} annotierte Entitäten (IoU threshold: {overlap_threshold})")
    else:
        print(f"\n[2/7] Keine Ground Truth gefunden ({annotations_file.name})")

    # Setup NLP engine (shared)
    print("\n[3/7] Lade SpaCy NLP engine...")
    nlp_engine = get_nlp_engine()
    if nlp_engine:
        print("      ✓ de_core_news_sm geladen")

    # Setup analyzers
    print("\n[4/7] Erstelle Analyzer...")
    print("      Pattern-Analyzer...")
    pattern_analyzer = setup_pattern_analyzer(nlp_engine)
    print("      ✓ Pattern-Analyzer bereit (8 deutsche Recognizer)")

    print("      NER-Analyzer (GLiNER)...")
    ner_analyzer = setup_ner_analyzer(nlp_engine)
    print("      ✓ NER-Analyzer bereit")

    print(f"      LLM-Analyzer ({model})...")
    llm_analyzer = setup_llm_analyzer(nlp_engine, model=model)
    print("      ✓ LLM-Analyzer bereit")

    # Analyze with each method
    print("\n[5/7] Analysiere Text...")

    # Helper to calculate metrics if ground truth exists
    def get_metrics(results: list) -> tuple[dict | None, dict | None]:
        """Returns (overall_metrics, domain_metrics_dict)."""
        if not ground_truth:
            return None, None

        detected = results_to_annotated_entities(results, text)
        overall = evaluate_results(detected, ground_truth, overlap_threshold=overlap_threshold)

        # Domain-specific metrics
        pattern_gt = [g for g in ground_truth if g.entity_type in PATTERN_DOMAIN]
        llm_gt = [g for g in ground_truth if g.entity_type in LLM_DOMAIN]
        pattern_det = [d for d in detected if d.entity_type in PATTERN_DOMAIN]
        llm_det = [d for d in detected if d.entity_type in LLM_DOMAIN]

        domain_metrics = {}
        if pattern_gt:
            domain_metrics["Pattern-Domain"] = evaluate_results(
                pattern_det, pattern_gt, overlap_threshold=overlap_threshold
            )
        if llm_gt:
            domain_metrics["LLM-Domain"] = evaluate_results(
                llm_det, llm_gt, overlap_threshold=overlap_threshold
            )

        return overall, domain_metrics

    print("\n      Pattern-Erkennung...")
    pattern_results = analyze_text(pattern_analyzer, text)
    pattern_metrics, pattern_domain = get_metrics(pattern_results)
    print_results_summary("Pattern", pattern_results, pattern_metrics, pattern_domain)

    print("\n      NER-Erkennung (GLiNER)...")
    ner_results = analyze_text(ner_analyzer, text)
    ner_metrics, ner_domain = get_metrics(ner_results)
    print_results_summary("NER", ner_results, ner_metrics, ner_domain)

    print("\n      LLM-Erkennung (Ministral)...")
    llm_results = analyze_text(llm_analyzer, text)
    llm_metrics, llm_domain = get_metrics(llm_results)
    print_results_summary("LLM", llm_results, llm_metrics, llm_domain)

    # Merge results (Pattern + LLM only, GLiNER excluded due to low precision)
    print("\n[6/7] Kombiniere Ergebnisse (Regex+LLM)...")
    combined_results = merge_results([pattern_results, llm_results])
    combined_metrics, combined_domain = get_metrics(combined_results)
    print_results_summary("Regex+LLM", combined_results, combined_metrics, combined_domain)

    # Anonymize
    print("\n[7/7] Anonymisiere Text...")
    anonymizer = AnonymizerEngine()
    operators = get_anonymization_operators()

    # Use deduplicated combined results for anonymization
    deduped_combined = remove_overlapping_entities(combined_results)
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=deduped_combined,
        operators=operators,
    )
    anonymized_text = anonymized.text

    # Save outputs
    output_dir = Path(__file__).parent / "ausgabe"
    output_dir.mkdir(exist_ok=True)

    # Build metrics dict for report (if ground truth available)
    metrics_for_report = None
    missed_legend = ""
    missed_highlighted = ""

    if ground_truth:
        metrics_for_report = {
            "Regex (Pattern)": pattern_metrics,
            "Nvidia-PII-GLiNER": ner_metrics,
            f"LLM ({model})": llm_metrics,
            "Regex+LLM": combined_metrics,
        }

        # Calculate missed entities (false negatives) for the combined approach
        detected_entities = results_to_annotated_entities(combined_results, text)
        missed_entities = get_missed_entities(detected_entities, ground_truth, overlap_threshold)

        # Convert missed entities to RecognizerResult-like format for highlighting
        from presidio_analyzer import RecognizerResult
        missed_as_results = [
            RecognizerResult(
                entity_type=e.entity_type,
                start=e.start,
                end=e.end,
                score=1.0,
            )
            for e in missed_entities
        ]
        missed_legend = build_legend(missed_as_results)
        missed_highlighted = build_highlighted_text(text, missed_as_results)

    # Generate and save comparison HTML report
    html_report = generate_report(
        input_filename=input_file.name,
        text_length=len(text),
        word_count=len(text.split()),
        pattern_count=len(remove_overlapping_entities(pattern_results)),
        ner_count=len(remove_overlapping_entities(ner_results)),
        llm_count=len(remove_overlapping_entities(llm_results)),
        combined_count=len(remove_overlapping_entities(combined_results)),
        pattern_legend=build_legend(pattern_results),
        ner_legend=build_legend(ner_results),
        llm_legend=build_legend(llm_results),
        combined_legend=build_legend(combined_results),
        pattern_highlighted=build_highlighted_text(text, pattern_results),
        ner_highlighted=build_highlighted_text(text, ner_results),
        llm_highlighted=build_highlighted_text(text, llm_results),
        combined_highlighted=build_highlighted_text(text, combined_results),
        anonymized_text=anonymized_text,
        original_text=text,
        metrics=metrics_for_report,
        missed_legend=missed_legend,
        missed_highlighted=missed_highlighted,
    )
    ## Include model name in output filename if not default - now chaneged always include it
    #if model != "ministral-3:8b":
    model_suffix = model.replace(":", "_").replace("-", "_")
    html_path = output_dir / f"demo_bericht_{model_suffix}.html"
    #else:
    #    html_path = output_dir / "demo_bericht.html"
    html_path.write_text(html_report, encoding="utf-8")

    print("\n" + "=" * 60)
    print("FERTIG")
    print("=" * 60)
    print(f"\n✓ HTML-Report: {html_path}")
    print(f"\nÖffne im Browser: file://{html_path.absolute()}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="German clinical text - method comparison")
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to input text file (default: data/entlassungsbrief.txt)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="IoU overlap threshold for matching (default: 0.5, try 0.3 for partial matches)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ministral-3:8b",
        help="Ollama model to use (default: ministral-3:8b, try ministral-3:14b for better recall)",
    )
    args = parser.parse_args()

    main(args.input_file, overlap_threshold=args.threshold, model=args.model)

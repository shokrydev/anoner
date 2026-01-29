#!/usr/bin/env python3
"""
Demo: Vergleich der Erkennungsmethoden für deutsche klinische Texte.

Zeigt separat: Pattern-only vs. NER-only vs. LLM-only vs. Kombiniert

Siehe README.md für Voraussetzungen und Verwendung.
"""

import html
import logging
import sys
import warnings
from datetime import datetime
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
    MinistralOllamaRecognizer,
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

# Default test input file
DEFAULT_INPUT_FILE = Path(__file__).parent / "eingabe" / "entlassungsbrief.txt"

# Entity types to detect
ENTITIES = [
    "PERSON", "LOCATION", "ORGANIZATION", "PHONE_NUMBER", "EMAIL_ADDRESS",
    "DATE_TIME", "AGE", "OCCUPATION", "ID",
    "DE_KVNR", "DE_LANR", "DE_BSNR", "DE_TELEMATIK_ID",
    "DE_PERSONAL_ID", "DE_TAX_ID", "DE_SOCIAL_SECURITY", "DE_POSTAL_CODE",
]

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


def setup_llm_analyzer(nlp_engine) -> AnalyzerEngine:
    """Set up analyzer with only MinistralOllamaRecognizer."""
    registry = RecognizerRegistry(supported_languages=["de"])

    try:
        ministral_recognizer = MinistralOllamaRecognizer(
            ollama_url="http://localhost:11434",
            model="ministral-3:8b",
            supported_language="de",
            timeout=120.0,
        )
        registry.add_recognizer(ministral_recognizer)
    except Exception as e:
        print(f"  ✗ MinistralOllamaRecognizer failed: {e}")

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


def generate_comparison_html_report(
    text: str,
    pattern_results: list,
    ner_results: list,
    llm_results: list,
    combined_results: list,
    anonymized_text: str,
    input_file: Path,
) -> str:
    """Generate HTML report comparing all recognition methods."""

    # Build highlighted texts
    pattern_highlighted = build_highlighted_text(text, pattern_results)
    ner_highlighted = build_highlighted_text(text, ner_results)
    llm_highlighted = build_highlighted_text(text, llm_results)
    combined_highlighted = build_highlighted_text(text, combined_results)

    # Build legends
    pattern_legend = build_legend(pattern_results)
    ner_legend = build_legend(ner_results)
    llm_legend = build_legend(llm_results)
    combined_legend = build_legend(combined_results)

    # Counts
    pattern_count = len(remove_overlapping_entities(pattern_results))
    ner_count = len(remove_overlapping_entities(ner_results))
    llm_count = len(remove_overlapping_entities(llm_results))
    combined_count = len(remove_overlapping_entities(combined_results))

    report_html = f'''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PII-Analyse Vergleich: {html.escape(input_file.name)}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f9fa;
            color: #212529;
        }}
        h1 {{
            color: #212529;
            border-bottom: 3px solid #228be6;
            padding-bottom: 10px;
        }}
        h2 {{ color: #495057; margin-top: 30px; }}
        .meta {{
            background: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .meta p {{ margin: 5px 0; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-card .number {{
            font-size: 1.8em;
            font-weight: bold;
            color: #228be6;
        }}
        .stat-card .label {{
            color: #868e96;
            font-size: 0.85em;
        }}
        .stat-card.pattern .number {{ color: #40c057; }}
        .stat-card.ner .number {{ color: #7950f2; }}
        .stat-card.llm .number {{ color: #fd7e14; }}
        .stat-card.combined .number {{ color: #228be6; }}
        .legend {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-height: 50px;
        }}
        .legend-item {{
            display: inline-flex;
            align-items: center;
            margin: 5px 15px 5px 0;
            font-size: 0.9em;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
            margin-right: 6px;
            border: 1px solid rgba(0,0,0,0.1);
        }}
        .no-results {{
            color: #868e96;
            font-style: italic;
        }}
        .text-panel {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            white-space: pre-wrap;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 0.9em;
            line-height: 1.8;
        }}
        .entity {{
            padding: 2px 4px;
            border-radius: 3px;
            cursor: help;
            border: 1px solid rgba(0,0,0,0.1);
        }}
        .tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-bottom: 0;
        }}
        .tab {{
            padding: 10px 15px;
            background: #e9ecef;
            border: none;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            font-size: 0.95em;
        }}
        .tab.active {{
            background: white;
            font-weight: bold;
        }}
        .tab.pattern {{ border-top: 3px solid #40c057; }}
        .tab.ner {{ border-top: 3px solid #7950f2; }}
        .tab.llm {{ border-top: 3px solid #fd7e14; }}
        .tab.combined {{ border-top: 3px solid #228be6; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .method-info {{
            background: #f1f3f4;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            font-size: 0.9em;
        }}
        footer {{
            text-align: center;
            color: #868e96;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
        }}
    </style>
</head>
<body>
    <h1>PII-Analyse Vergleich</h1>

    <div class="meta">
        <p><strong>Eingabedatei:</strong> {html.escape(str(input_file.name))}</p>
        <p><strong>Erstellt:</strong> {datetime.now().strftime("%d.%m.%Y %H:%M:%S")}</p>
        <p><strong>Textlänge:</strong> {len(text):,} Zeichen, ~{len(text.split()):,} Wörter</p>
    </div>

    <h2>Erkannte Entitäten pro Methode</h2>
    <div class="stats">
        <div class="stat-card combined">
            <div class="number">{combined_count}</div>
            <div class="label">Kombiniert</div>
        </div>
        <div class="stat-card pattern">
            <div class="number">{pattern_count}</div>
            <div class="label">Regex</div>
        </div>
        <div class="stat-card ner">
            <div class="number">{ner_count}</div>
            <div class="label">Nvidia-PII-GLiNER</div>
        </div>
        <div class="stat-card llm">
            <div class="number">{llm_count}</div>
            <div class="label">LLM (Ministral)</div>
        </div>
    </div>

    <h2>Ergebnisse</h2>
    <div class="tabs">
        <button class="tab combined active" onclick="showTab('combined')">Kombiniert</button>
        <button class="tab pattern" onclick="showTab('pattern')">Regex (Kennnummern)</button>
        <button class="tab ner" onclick="showTab('ner')">Nvidia-PII-GLiNER</button>
        <button class="tab llm" onclick="showTab('llm')">LLM (Ministral)</button>
        <button class="tab" onclick="showTab('anonymized')">Anonymisiert</button>
        <button class="tab" onclick="showTab('original')">Original</button>
    </div>

    <div id="combined" class="tab-content active">
        <div class="method-info">
            <strong>Kombiniert:</strong> Kombination aller drei Methoden
            (bei Überlappung wird höchster Score behalten)
        </div>
        <div class="legend">{combined_legend}</div>
        <div class="text-panel">{combined_highlighted}</div>
    </div>

    <div id="pattern" class="tab-content">
        <div class="method-info">
            <strong>Regex (Kennnummern):</strong> Regex-basierte Erkennung deutscher Kennnummern
            (KVNR, LANR, BSNR, Telematik-ID, Personalausweis, Steuer-ID, SVNR, PLZ)
        </div>
        <div class="legend">{pattern_legend}</div>
        <div class="text-panel">{pattern_highlighted}</div>
    </div>

    <div id="ner" class="tab-content">
        <div class="method-info">
            <strong>Nvidia-PII-GLiNER:</strong> Zero-shot Transformer-Encoder-basierte PII-Erkennung
            (Tokenweise Klassifikation)
        </div>
        <div class="legend">{ner_legend}</div>
        <div class="text-panel">{ner_highlighted}</div>
    </div>

    <div id="llm" class="tab-content">
        <div class="method-info">
            <strong>MinistralOllamaRecognizer:</strong> LLM-basierte PII-Extraktion via Ministral 8B
            (lokal gehostet mit Ollama)
        </div>
        <div class="legend">{llm_legend}</div>
        <div class="text-panel">{llm_highlighted}</div>
    </div>

    <div id="anonymized" class="tab-content">
        <div class="method-info">
            <strong>Anonymisiert:</strong> Text mit ersetzten PII-Entitäten (basierend auf kombinierter Erkennung)
        </div>
        <div class="text-panel">{html.escape(anonymized_text)}</div>
    </div>

    <div id="original" class="tab-content">
        <div class="method-info">
            <strong>Original:</strong> Unverarbeiteter Eingabetext
        </div>
        <div class="text-panel">{html.escape(text)}</div>
    </div>

    <footer>
        Generiert mit Presidio Anonymizer - Methodenvergleich
    </footer>

    <script>
        function showTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>'''

    return report_html


def print_results_summary(name: str, results: list) -> None:
    """Print a summary of detected entities."""
    filtered = remove_overlapping_entities(results)
    by_type = {}
    for r in filtered:
        if r.entity_type not in by_type:
            by_type[r.entity_type] = []
        by_type[r.entity_type].append(r)

    print(f"\n  {name}: {len(filtered)} Entitäten")
    for entity_type, entities in sorted(by_type.items()):
        print(f"    - {entity_type}: {len(entities)}")


def main(input_file: Path = DEFAULT_INPUT_FILE):
    """Main function to run the comparison pipeline."""
    print("\n" + "=" * 60)
    print("GERMAN CLINICAL TEXT - METHODEN-VERGLEICH")
    print("=" * 60)

    # Load input text
    print(f"\n[1/6] Lade Text: {input_file.name}")
    text = load_input_text(input_file)
    print(f"      {len(text)} Zeichen, ~{len(text.split())} Wörter")

    # Setup NLP engine (shared)
    print("\n[2/6] Lade SpaCy NLP engine...")
    nlp_engine = get_nlp_engine()
    if nlp_engine:
        print("      ✓ de_core_news_sm geladen")

    # Setup analyzers
    print("\n[3/6] Erstelle Analyzer...")
    print("      Pattern-Analyzer...")
    pattern_analyzer = setup_pattern_analyzer(nlp_engine)
    print("      ✓ Pattern-Analyzer bereit (8 deutsche Recognizer)")

    print("      NER-Analyzer (GLiNER)...")
    ner_analyzer = setup_ner_analyzer(nlp_engine)
    print("      ✓ NER-Analyzer bereit")

    print("      LLM-Analyzer (Ministral)...")
    llm_analyzer = setup_llm_analyzer(nlp_engine)
    print("      ✓ LLM-Analyzer bereit")

    # Analyze with each method
    print("\n[4/6] Analysiere Text...")

    print("\n      Pattern-Erkennung...")
    pattern_results = analyze_text(pattern_analyzer, text)
    print_results_summary("Pattern", pattern_results)

    print("\n      NER-Erkennung (GLiNER)...")
    ner_results = analyze_text(ner_analyzer, text)
    print_results_summary("NER", ner_results)

    print("\n      LLM-Erkennung (Ministral)...")
    llm_results = analyze_text(llm_analyzer, text)
    print_results_summary("LLM", llm_results)

    # Merge results
    print("\n[5/6] Kombiniere Ergebnisse...")
    combined_results = merge_results([pattern_results, ner_results, llm_results])
    print_results_summary("Kombiniert", combined_results)

    # Anonymize
    print("\n[6/6] Anonymisiere Text...")
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

    # Generate and save comparison HTML report
    html_report = generate_comparison_html_report(
        text=text,
        pattern_results=pattern_results,
        ner_results=ner_results,
        llm_results=llm_results,
        combined_results=combined_results,
        anonymized_text=anonymized_text,
        input_file=input_file,
    )
    html_path = output_dir / "demo_bericht.html"
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
    args = parser.parse_args()

    main(args.input_file)

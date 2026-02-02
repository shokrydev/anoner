"""
HTML report template for PII detection comparison.

Extracted from demo_skript.py for maintainability.
"""

import html
from datetime import datetime


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PII-Analyse Vergleich: {input_filename}</title>
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
        .tab.missed {{ border-top: 3px solid #e03131; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        .method-info {{
            background: #f1f3f4;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            font-size: 0.9em;
        }}
        .metrics-section {{
            background: #e7f5ff;
            border: 1px solid #74c0fc;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .metrics-section h3 {{
            margin-top: 0;
            color: #1971c2;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 5px;
            overflow: hidden;
        }}
        .metrics-table th, .metrics-table td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .metrics-table th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        .metrics-table tr:last-child td {{
            border-bottom: none;
        }}
        .metric-value {{
            font-family: "SFMono-Regular", Consolas, monospace;
            font-weight: 600;
        }}
        .metric-good {{ color: #2f9e44; }}
        .metric-medium {{ color: #f59f00; }}
        .metric-poor {{ color: #e03131; }}
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
        <p><strong>Eingabedatei:</strong> {input_filename}</p>
        <p><strong>Erstellt:</strong> {timestamp}</p>
        <p><strong>Textlänge:</strong> {text_length:,} Zeichen, ~{word_count:,} Wörter</p>
    </div>

    <h2>Erkannte Entitäten pro Methode</h2>
    <div class="stats">
        <div class="stat-card combined">
            <div class="number">{combined_count}</div>
            <div class="label">Regex+LLM</div>
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

    {metrics_section}

    <h2>Ergebnisse</h2>
    <div class="tabs">
        <button class="tab combined active" onclick="showTab('combined')">Regex+LLM</button>
        <button class="tab pattern" onclick="showTab('pattern')">Regex (Kennnummern)</button>
        <button class="tab ner" onclick="showTab('ner')">Nvidia-PII-GLiNER</button>
        <button class="tab llm" onclick="showTab('llm')">LLM (Ministral)</button>
        <button class="tab missed" onclick="showTab('missed')">Verpasst</button>
        <button class="tab" onclick="showTab('anonymized')">Anonymisiert</button>
        <button class="tab" onclick="showTab('original')">Original</button>
    </div>

    <div id="combined" class="tab-content active">
        <div class="method-info">
            <strong>Regex+LLM:</strong> Kombination aus Regex-Pattern und LLM-Erkennung
            (GLiNER ausgeschlossen wegen niedriger Precision; bei Überlappung wird höchster Score behalten)
        </div>
        <div class="legend">{combined_legend}</div>
        <div class="text-panel">{combined_highlighted}</div>
    </div>

    <div id="missed" class="tab-content">
        <div class="method-info">
            <strong>Verpasst (False Negatives):</strong> Entitäten aus der Ground Truth, die von Regex+LLM nicht erkannt wurden.
            Diese Lücken sind das Ziel für Verbesserungen.
        </div>
        <div class="legend">{missed_legend}</div>
        <div class="text-panel">{missed_highlighted}</div>
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
            <strong>OllamaNERecognizer:</strong> LLM-basierte PII-Extraktion via Ministral 8B
            (lokal gehostet mit Ollama)
        </div>
        <div class="legend">{llm_legend}</div>
        <div class="text-panel">{llm_highlighted}</div>
    </div>

    <div id="anonymized" class="tab-content">
        <div class="method-info">
            <strong>Anonymisiert:</strong> Text mit ersetzten PII-Entitäten (basierend auf Regex+LLM Erkennung)
        </div>
        <div class="text-panel">{anonymized_text}</div>
    </div>

    <div id="original" class="tab-content">
        <div class="method-info">
            <strong>Original:</strong> Unverarbeiteter Eingabetext
        </div>
        <div class="text-panel">{original_text}</div>
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


def _get_metric_class(value: float) -> str:
    """Return CSS class based on metric value."""
    if value >= 0.8:
        return "metric-good"
    elif value >= 0.5:
        return "metric-medium"
    else:
        return "metric-poor"


def build_metrics_section(metrics: dict[str, dict] | None) -> str:
    """
    Build the metrics section HTML.

    Args:
        metrics: Dictionary mapping method names to their metrics dict,
                 or None if no ground truth is available.

    Returns:
        HTML string for the metrics section.
    """
    if not metrics:
        return ""

    rows = []
    for method, m in metrics.items():
        p_class = _get_metric_class(m["precision"])
        r_class = _get_metric_class(m["recall"])
        f_class = _get_metric_class(m["f_beta"])

        rows.append(f'''        <tr>
            <td>{html.escape(method)}</td>
            <td class="metric-value">{m["tp"]}</td>
            <td class="metric-value">{m["fp"]}</td>
            <td class="metric-value">{m["fn"]}</td>
            <td class="metric-value {p_class}">{m["precision"]:.1%}</td>
            <td class="metric-value {r_class}">{m["recall"]:.1%}</td>
            <td class="metric-value {f_class}">{m["f_beta"]:.1%}</td>
        </tr>''')

    return f'''<div class="metrics-section">
        <h3>Evaluation gegen Ground Truth</h3>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Methode</th>
                    <th>TP</th>
                    <th>FP</th>
                    <th>FN</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F2-Score</th>
                </tr>
            </thead>
            <tbody>
{chr(10).join(rows)}
            </tbody>
        </table>
    </div>'''


def generate_report(
    input_filename: str,
    text_length: int,
    word_count: int,
    pattern_count: int,
    ner_count: int,
    llm_count: int,
    combined_count: int,
    pattern_legend: str,
    ner_legend: str,
    llm_legend: str,
    combined_legend: str,
    pattern_highlighted: str,
    ner_highlighted: str,
    llm_highlighted: str,
    combined_highlighted: str,
    anonymized_text: str,
    original_text: str,
    metrics: dict[str, dict] | None = None,
    missed_legend: str = "",
    missed_highlighted: str = "",
) -> str:
    """
    Generate the HTML report from template.

    Args:
        input_filename: Name of the input file.
        text_length: Length of text in characters.
        word_count: Approximate word count.
        pattern_count: Number of pattern-detected entities.
        ner_count: Number of NER-detected entities.
        llm_count: Number of LLM-detected entities.
        combined_count: Number of combined entities.
        pattern_legend: HTML for pattern legend.
        ner_legend: HTML for NER legend.
        llm_legend: HTML for LLM legend.
        combined_legend: HTML for combined legend.
        pattern_highlighted: HTML with highlighted pattern entities.
        ner_highlighted: HTML with highlighted NER entities.
        llm_highlighted: HTML with highlighted LLM entities.
        combined_highlighted: HTML with highlighted combined entities.
        anonymized_text: Anonymized text (will be escaped).
        original_text: Original text (will be escaped).
        metrics: Optional dict of method name -> metrics dict for evaluation section.
        missed_legend: HTML for missed entities legend (false negatives).
        missed_highlighted: HTML with highlighted missed entities.

    Returns:
        Complete HTML report string.
    """
    return HTML_TEMPLATE.format(
        input_filename=html.escape(input_filename),
        timestamp=datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        text_length=text_length,
        word_count=word_count,
        pattern_count=pattern_count,
        ner_count=ner_count,
        llm_count=llm_count,
        combined_count=combined_count,
        pattern_legend=pattern_legend,
        ner_legend=ner_legend,
        llm_legend=llm_legend,
        combined_legend=combined_legend,
        pattern_highlighted=pattern_highlighted,
        ner_highlighted=ner_highlighted,
        llm_highlighted=llm_highlighted,
        combined_highlighted=combined_highlighted,
        anonymized_text=html.escape(anonymized_text),
        original_text=html.escape(original_text),
        metrics_section=build_metrics_section(metrics),
        missed_legend=missed_legend if missed_legend else '<span class="no-results">Keine Ground Truth verfügbar</span>',
        missed_highlighted=missed_highlighted if missed_highlighted else html.escape(original_text),
    )

# Deutsche Demo: Anonymisierung klinischer Texte

Demo-Skript zum Testen und Vergleichen verschiedener Erkennungsmethoden für PII in deutschen klinischen Texten.

> **[📊 Demo-Report (8b)](https://htmlpreview.github.io/?https://github.com/shokrydev/anoner/blob/main/deutsche-demo/ausgabe/demo_bericht_ministral_3_8b.html)** | **[📊 Demo-Report (14b)](https://htmlpreview.github.io/?https://github.com/shokrydev/anoner/blob/main/deutsche-demo/ausgabe/demo_bericht_ministral_3_14b.html)** - Interaktive HTML-Reports mit Methodenvergleich
>
> **[📝 Findings & Learnings](ausgabe/README.md)** - Dokumentierte Erkenntnisse aus der Entwicklung

## Pipeline-Komponenten

**Erkennungsmethoden:**
- **Nvidia-PII-GLiNER** - Zero-shot Transformer-basierte PII-Erkennung (55+ Entitätstypen)
- **OllamaNERecognizer** - LLM-basierte PII-Extraktion via Ollama
- **Regex (Kennnummern)** - Pattern-basierte Erkennung deutscher Identifier

**Deutsche Pattern-Recognizer:**
| Recognizer | Entität | Beschreibung |
|------------|---------|--------------|
| KVNR | `DE_KVNR` | Krankenversichertennummer |
| Telematik-ID | `DE_TELEMATIK_ID` | Gesundheits-ID (10-) und eHBA (1-) |
| LANR | `DE_LANR` | Lebenslange Arztnummer |
| BSNR | `DE_BSNR` | Betriebsstättennummer |
| Steuer-ID | `DE_TAX_ID` | Steueridentifikationsnummer |
| SVNR | `DE_SOCIAL_SECURITY` | Sozialversicherungsnummer |
| Personalausweis | `DE_PERSONAL_ID` | Personalausweisnummer |
| PLZ | `DE_POSTAL_CODE` | Postleitzahl (Kontextabhängig) |

## Voraussetzungen

```bash
# Ollama starten und Modell herunterladen
systemctl start ollama  # oder: ollama serve
ollama pull ministral-3:8b

# Abhängigkeiten installieren (vom deutsche-demo-Verzeichnis aus)
cd ../presidio-analyzer && poetry install && cd ../deutsche-demo
```

## Verwendung

```bash
# Mit Standard-Testdatei ausführen (vom deutsche-demo-Verzeichnis aus)
cd ../presidio-analyzer && poetry run python ../deutsche-demo/demo_skript.py && cd ../deutsche-demo

# Mit 14b Ministral statt 8b (längere Ausführungszeit)
cd ../presidio-analyzer && poetry run python ../deutsche-demo/demo_skript.py --model ministral-3:14b && cd ../deutsche-demo

# Mit eigener Eingabedatei ausführen
cd ../presidio-analyzer && poetry run python ../deutsche-demo/demo_skript.py /pfad/zu/datei.txt && cd ../deutsche-demo
```

## Dateien

**Eingabe:**
- `eingabe/entlassungsbrief.txt` - Beispiel eines deutschen Entlassungsbriefs (~770 Wörter)

**Ausgabe (nach Durchlauf):**
- `ausgabe/demo_bericht_ministral_3_8b.html` - HTML-Report (8b Modell)
- `ausgabe/demo_bericht_ministral_3_14b.html` - HTML-Report (14b Modell)

**HTML-Report Tabs:**
| Tab | Beschreibung |
|-----|--------------|
| Regex+LLM | Kombination aus Pattern und LLM (Default-Ansicht) |
| Regex (Kennnummern) | Nur Pattern-basierte Erkennung |
| Nvidia-PII-GLiNER | Nur NER-basierte Erkennung |
| LLM (Ministral) | Nur LLM-basierte Erkennung |
| Verpasst | Entitäten die nicht erkannt wurden (False Negatives) |
| Anonymisiert | Text mit ersetzten PII-Entitäten |
| Original | Unverarbeiteter Eingabetext |

## Evaluation

**Erreichte Ergebnisse** (auf synthetischem Entlassungsbrief mit 71 annotierten Entitäten):

| Methode | Precision | Recall | F2-Score |
|---------|-----------|--------|----------|
| Pattern-only | 93% | 20% | 23% |
| GLiNER-only | 45% | 66% | 60% |
| **Regex+LLM (8b)** | **95%** | **78%** | **80%** |
| **Regex+LLM (14b)** | **100%** | **76%** | **80%** |

**Modellvergleich:**
- 8b-Modell: Höherer Recall (78%), gute Precision (95%)
- 14b-Modell: Perfekte Precision (100%), etwas geringerer Recall (76%)
- LLM-Ausgaben können zwischen Durchläufen variieren

**Domänenspezifisch:**
- Pattern-Domain (DE_* IDs): P=100% R=100% F2=100%

GLiNER wurde aus der Kombination entfernt (hohe False-Positive-Rate ohne Mehrwert).

### Potenzielle annotierte Datensätze

Für Evaluation auf echten klinischen Daten:

| Datensatz | Beschreibung | Zugang |
|-----------|--------------|--------|
| **[GGPONC 2.0](https://www.leitlinienprogramm-onkologie.de/projekte/ggponc-deutsch)** | 30 deutsche Onkologie-Leitlinien, 1.87M Tokens. Annotiert mit Diagnosen, Substanzen, Prozeduren. Kein PHI (Leitlinientexte). | Zugangsanfrage via [Zenodo](https://zenodo.org/communities/german-cancer-society/records?f=subject:ggponc) mit Projektbeschreibung. Zitierung erforderlich. |
| **[BRONCO150](https://www2.informatik.hu-berlin.de/~leser/bronco/index.html)** | 150 deutsche Entlassungsbriefe (Onkologie). Sätze randomisiert zur Anonymisierung. Annotiert mit med. Entitäten und Attributen (Negation, Spekulation). | Data Usage Agreement (DUA) an Prof. Ulf Leser (HU Berlin). Nur für akademische Forschung. |

**Hinweis:** Beide Datensätze sind nicht frei verfügbar, sondern erfordern eine Zugangsanfrage bzw. Nutzungsvereinbarung.

## Hinweise zur Performance

Nvidia-PII-GLiNER und Ministral 8B nutzen beide die GPU. Bei 12GB VRAM kann es zu Speicherkonflikten kommen, die zu Modell-Neuladen führen. Bei langsamer Performance:

1. Kleineres Ollama-Modell verwenden (z.B. `mistral:7b`)
2. GLiNER auf CPU zwingen: `map_location="cpu"` im Skript hinzufügen

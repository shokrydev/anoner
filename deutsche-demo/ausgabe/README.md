# Findings und Learnings zur PII-Erkennung in klinischen Texten

Dieses Dokument dokumentiert die praktische Erprobung von PII-Erkennung in deutschen klinischen Texten:

- **Teil 1**: Empirische Beobachtungen (Findings)
- **Teil 2**: Abgeleitete Erkenntnisse (Learnings)
- **Teil 3**: Quantitative Evaluation mit Metriken
- **Teil 4**: Motivation für einen agentenbasierten Ansatz

---

# Teil 1: Findings

Konkrete Beobachtungen aus der Analyse eines deutschen Entlassungsbriefs mit drei Erkennungsmethoden (Regex, GLiNER, LLM).

## False Positives

### Medizinische Begriffe als Entitäten

| Erkannt als | Text | Score | Problem |
|-------------|------|-------|---------|
| PERSON | "Nikotinkarenz" | 0.99 | Medizinischer Fachbegriff fälschlich als Name |
| PERSON | "re alt" | 0.46 | Tokenisierungsfehler bei "82 Jahre alt" |
| ORGANIZATION | "Mediterrane Diät" | 0.83 | Ernährungsempfehlung als Institution |
| ORGANIZATION | "salzarme Kost" | 0.53 | Ernährungsempfehlung als Institution |
| LOCATION | "Fluss" in "TIMI-3-Fluss" | 0.66 | Kardiologischer Fachbegriff (Blutfluss-Klassifikation) |

### Rollenbezeichnungen als Beruf

| Erkannt als | Text | Score | Problem |
|-------------|------|-------|---------|
| OCCUPATION | "Patient" | 0.49-0.94 | Mehrfach im Text, keine Berufsbezeichnung |
| OCCUPATION | "Herzkatheterlabor" | 0.66 | Räumlichkeit, kein Beruf |
| OCCUPATION | "Dr. med." | 0.99 | Akademischer Grad, kein Beruf |

Das Wort "Patient" erscheint 8x im Text und wird von GLiNER jedes Mal als BERUF erkannt. Nach Anonymisierung steht dort "[BERUF] wurde aufgeklärt" statt "Patient wurde aufgeklärt". Der Text wird unlesbar.

## Attention Competition bei Transformer-Modellen

GLiNER hat begrenzte Aufmerksamkeitskapazität pro Chunk. Frühere Entitäten unterdrücken die Erkennung späterer ähnlicher Entitäten.

| Context-Start | "58-jähriger" erkannt? | Was sich ändert |
|---------------|------------------------|-----------------|
| 2165 | Ja | Nur der Zielsatz |
| 2100 | Ja | Kleiner Prefix |
| 2050 | Nein | Jetzt kommt "29 Jahre" davor |
| 2000 | Nein | Mehr Altersangaben davor |
| 1500 | Nein | Noch mehr Alter (32, 62, 82) |

Ab Position 2050 enthält der Text "29 Jahre" vor "58-jähriger". Die frühere Altersangabe wird mit Score 0.96 erkannt und verhindert die Erkennung der späteren. Das ist kein Chunk-Boundary-Problem sondern passiert innerhalb des Chunks.

Klinische Dokumente enthalten dichte wiederholte Entity-Typen:
- Mehrere Familienalter in der Familienanamnese
- Mehrere Datumsangaben (Geburt, Aufnahme, Eingriffe, Entlassung)
- Mehrere Namen (Patient, Ärzte, Familie, Kontakte)

Erste Vorkommen werden erkannt, spätere fehlen. Die Anonymisierung wird inkonsistent.

## Kontextabhängige Entity-Klassifikation

Dasselbe Textmuster wird je nach umgebendem Dokumentinhalt unterschiedlich klassifiziert.

```
# Isolierter Test:
"E-Mail: max.mustermann@email.de" → EMAIL_ADDRESS (Score 1.0)

# Im vollen Dokument mit vielen Namen:
"E-Mail: max.mustermann@email.de" → PERSON "max" + PERSON "mustermann"
                                    EMAIL_ADDRESS NICHT erkannt
```

In Dokumenten mit vielen Namensentitäten tendiert GLiNER dazu, namensähnliche Strings als PERSON zu klassifizieren. Das lokale Muster "max.mustermann" sieht aus wie zwei Namen und der Dokumentkontext verstärkt das.

Ergebnis: `[PERSON].[PERSON]@email.de` - partielle Anonymisierung die schlimmer ist als keine (offenbart Email-Struktur ohne sie zu schützen).

## Gleiches Wort, verschiedene Bedeutung

| Text | Was anonymisieren? | Warum |
|------|-------------------|-------|
| "Behandelnder Arzt: Dr. Müller" | Nur "Dr. Müller" | "Behandelnder Arzt" ist Label, Name ist PII |
| "Der Patient arbeitet als Arzt" | "Arzt" | Beruf des Patienten (Quasi-Identifier) |
| "Pflegekraft notierte..." | Nichts | Rolle im Dokument |
| "Seine Frau ist Pflegekraft" | "Pflegekraft" | Beruf des Familienmitglieds |

"Arzt", "Krankenschwester", "Pfleger" sind sowohl medizinische Rollen als auch potenzielle Berufe von Patient oder Familie. Eine einfache Deny-Liste würde Patientenberufe verpassen. Alles erkennen würde die Dokumentstruktur zerstören.

## Grenzfälle

### Feldbezeichnungen vs. Werte

"Aufnahmedatum" und "Geburtsdatum" werden als DATE_TIME erkannt obwohl es Labels sind. Das eigentliche Datum dahinter (05.01.2026, 15.03.1965) wird separat erkannt. Die Labels zu anonymisieren macht den Text unverständlich:

```
Vorher:  Geburtsdatum: 15.03.1965
Nachher: [DATUM]: [DATUM]
```

### Relative Zeitangaben

"3. postinterventionellen Tag" wird als DATE_TIME erkannt. Diese Information identifiziert niemanden. Anonymisierung entfernt klinisch relevante Verlaufsinformationen ohne Datenschutzgewinn.

### Berufsbezeichnungen

| Text | Kontext | Anonymisieren? |
|------|---------|----------------|
| "Bankangestellter" | Sozialanamnese des Patienten | Diskutabel |
| "Oberarzt Kardiologie" | Unterschrift des Arztes | Nein (öffentlich bekannt) |
| "Hausarzt" | Verteiler-Liste | Nein (Rollenbezeichnung) |

Die Berufsbezeichnung des Patienten ("Bankangestellter") könnte theoretisch zur Re-Identifikation beitragen. Die Berufe der behandelnden Ärzte sind hingegen öffentlich zugänglich und deren Anonymisierung sinnlos.

## Tokenisierungsprobleme

Bei "82 Jahre alt" wurde die Zahl 82 korrekt als AGE erkannt. Der Rest "re alt" wurde dann fälschlich als PERSON klassifiziert. Dies zeigt dass die Modelle auf Tokenebene arbeiten und der Kontext zwischen benachbarten Tokens verloren gehen kann.

## Presidio-Sprachunterstützung

Manche "universellen" Recognizer unterstützen nur bestimmte Sprachen:

```python
# EmailRecognizer unterstützt nur Englisch
registry.get_recognizers(language='de', entities=['EMAIL_ADDRESS'])
→ ValueError: No matching recognizers found
```

EmailRecognizer nutzt englische Kontextwörter ("email:", "e-mail:") für Scoring. Deutsche Kontexte ("E-Mail:", "elektronische Post:") werden nicht unterstützt.

## Pattern-Recognizer: Deterministische Zuverlässigkeit

Alle pattern-basierten Recognizer (KVNR, Steuer-ID, Personalausweis, LANR, BSNR) erkannten mit Score 1.0 in den Testläufen.

Regex + Prüfsummenvalidierung ist deterministisch: Wenn ein Pattern mit gültiger Prüfsumme matched, ist es korrekt. Die Gesamt-Precision von ~93% entsteht durch Überlappungen zwischen Pattern-Typen, nicht durch falsche Erkennungen.

Aber: Selbst zwischen fest definierten Kennummern verschiedener Art können Überlappungen und Konflikte entstehen wenn die Pattern nicht sorgfältig abgegrenzt sind.

## LLM Pretraining Bias

Die Erkennungsleistung von LLMs hängt von Trainingssprache und -domäne ab. Ein auf englischen Texten trainiertes Modell erkennt deutsche Muster schlechter. Ein auf allgemeinen Texten trainiertes Modell kennt klinische Fachbegriffe nicht.

Naives Prompting mit der Forderung "erkenne personenbezogene Daten" reicht nicht. Sekundäre Identifikatoren werden ignoriert wenn sie nicht explizit genannt werden. Auch bei der Technik mit der stärksten Generalisierbarkeit muss im Vorhinein bekannt sein in welcher Form zu anonymisierende Daten vorkommen können.

## Identifikationskraft variiert nach Kontext

Nicht alle Berufsbezeichnungen sind gleich:

| Beruf | Im klinischen Text | Identifikationskraft |
|-------|-------------------|---------------------|
| "Arzt" | Kommt ständig vor | Fast null (zu häufig) |
| "Bundeskanzler" | Extrem selten | Fast direkter Identifier |
| "Bankangestellter" | Selten | Mittel (Quasi-Identifier) |

Ein Modell das blind alle OCCUPATION-Entities anonymisiert behandelt "Arzt" und "Bundeskanzler" gleich obwohl ihre Re-Identifikationsrisiken völlig unterschiedlich sind.

## Deutsche Healthcare-IDs sind verkettet

Die deutschen Gesundheits-Identifikatoren bilden ein Netzwerk:

- **KVNR** (Patient) → behandelt von → **LANR** (Arzt)
- **LANR** (Arzt) → arbeitet bei → **BSNR** (Praxis/Klinik)
- **Telematik-ID** verbindet Patient (10-) und Arzt (1-) im digitalen System

Wenn eine ID bekannt ist können die anderen potenziell abgeleitet werden. Ein Entlassungsbrief enthält typischerweise alle drei.

## Klinische Dokumentstruktur

Verschiedene Sektionen enthalten verschiedene Entity-Typen:

| Sektion | Typische Inhalte |
|---------|------------------|
| Header | Arzt-IDs, Klinik, Kontaktdaten |
| Patientendaten | KVNR, Adresse, Geburtsdatum |
| Anamnese | Familiengeschichte, Alter, Berufe |
| Diagnosen | ICD-Codes, medizinische Begriffe |
| Therapie | Prozeduren, Medikamente, Daten |
| Entlassung | Reha-Einrichtungen, Folgetermine |

Eine sektionsbewusste Verarbeitung könnte unterschiedliche Regeln pro Abschnitt anwenden.

## Jedes klinische Dokument ist Hochrisiko

Klinische Dokumente enthalten immer sowohl direkte als auch Quasi-Identifikatoren. Es gibt keine "harmlosen" klinischen Texte.

**Daten sind überall**: Geburt, Aufnahme, Entlassung, Prozedur, Nachsorge. Ein einzelner Entlassungsbrief enthält leicht 10+ Datumsangaben.

**Namen erscheinen an unerwarteten Stellen**: In Email-Adressen (max.mustermann@...), Dateipfaden, Unterschriften, Verteiler-Listen. Ein Recognizer der nur nach dem Pattern "Name:" sucht verpasst diese.

## ICD-Codes als Quasi-Identifier

Diagnosecodes wie "I21.1" (STEMI Hinterwand) sind standardisiert und scheinen harmlos. Aber:

- Seltene Erkrankungen haben seltene Codes
- Code + Datum + Region kann eindeutig sein
- Kombinationen von Codes bilden Fingerprints

Ein Patient mit drei seltenen Komorbiditäten in einer bestimmten Klinik an einem bestimmten Tag ist potenziell identifizierbar auch ohne Namen.

## False Negatives: Was komplett verpasst wurde

Nicht nur False Positives sind problematisch. Folgende Entities wurden von keinem Recognizer erkannt:

| Verpasst | Text | Warum problematisch |
|----------|------|---------------------|
| Buchungsnummer | "MT-2026-4711" | Krankentransport-ID, verknüpft mit Patient |
| Stent-Modell | "Xience Pro 3,5 x 28mm" | Spezifisches Produkt + Maße kann identifizierend sein |
| Station | "Innere Medizin 3A" | Kombiniert mit Datum engt Population ein |

Die Kombination aus Stent-Typ, Implantationsdatum und Klinik könnte ausreichen um einen Patienten zu identifizieren.

---

# Teil 2: Learnings

Was wir aus den Findings für die Entwicklung von Anonymisierungsmethoden gelernt haben.

## DSGVO: Direkte vs. Quasi-Identifikatoren

**Direkte Identifikatoren** (identifizieren eine Person eindeutig):
- Namen (PERSON), Adressen (LOCATION), Telefon/Email
- Deutsche IDs: KVNR, Steuer-ID, Personalausweis, SVNR, Telematik-ID

**Quasi-Identifikatoren** (ermöglichen Re-Identifikation durch Kombination):
- AGE: "58-jährige Patient" kombiniert mit Diagnose engt Population ein
- OCCUPATION: "Bankangestellter" engt weiter ein
- Daten: Aufnahme-/Entlassungsdatum + Diagnose = einzigartiger Fingerprint
- Familiendetails: "Vater verstarb mit 62 an Herzinfarkt"
- Regionale Info: Spezifische Klinik + Reha-Einrichtung
- Referenznummern: Transportbuchung "MT-2026-4711"

Studien zur Re-Identifikation zeigen, dass oft 3-4 Quasi-Identifikatoren ausreichen. Ein klinischer Brief enthält typischerweise 10+.

| Risikofaktor | Beispiel im klinischen Text | Mitigation |
|--------------|----------------------------|------------|
| Seltene Erkrankung + Datum | STEMI am 15.03.2026 | Datum auf Monat/Jahr generalisieren |
| Spezifische Prozedur | "Xience Alpine DES Stent" | Markennamen entfernen |
| Exaktes Alter + Ort | "58 Jahre, München" | Altersgruppen, nur Region |
| Familienmedizingeschichte | Einzigartige Konstellation | Generalisieren oder entfernen |
| Buchungs-/Referenznummern | Transport-ID | Immer entfernen |

## Recognizer-Typen im Vergleich

| Typ | Stärken | Schwächen | Fazit |
|-----|---------|-----------|-------|
| Pattern | Deterministisch, prüfsummen-validiert | Nur strukturierte Formate | **Empfohlen** für deutsche IDs |
| GLiNER | Zero-shot, viele Entity-Typen | Viele False Positives, Attention Competition | **Nicht empfohlen** (siehe Teil 3) |
| LLM | Kontextverständnis, hohe Precision | Moderater Recall, langsamer | **Empfohlen** für Namen, Orte |
| Transformer NER | Domänenspezifisch wenn finetuned | Braucht gelabelte Trainingsdaten | Potentiell nach Finetuning |

Designprinzip: Pattern + LLM kombinieren. GLiNER bringt keinen Mehrwert (zu viele False Positives).

## Over-Anonymization

Jede fälschlich anonymisierte Stelle reduziert den Informationsgehalt. Bei klinischen Texten geht medizinisch relevanter Inhalt verloren:

- Ernährungsempfehlungen verschwinden
- Zeitliche Verläufe werden unkenntlich
- Fachbegriffe werden entstellt

Recall (alle PII finden) muss gegen Precision abgewogen werden. Ein False Positive der die Dokumentstruktur zerstört kann schlimmer sein als ein False Negative.

**Kritisch für Forschungsdaten**: Wenn alle theoretisch möglichen Quasi-Identifikatoren anonymisiert werden ist der Lehrgehalt der Daten für das Trainieren spezialisierter Modelle nichtig. Ein klinisches NER-Modell kann nicht auf Texten trainiert werden in denen alle medizinisch relevanten Begriffe durch Platzhalter ersetzt wurden.

## Domänenspezifische Ausschlusslisten

Für klinische Texte wären Ausschlusslisten sinnvoll:
- Medizinische Fachbegriffe (TIMI-Fluss, Nikotinkarenz)
- Strukturelle Begriffe (Patient, Aufnahmedatum)
- Rollenbezeichnungen im klinischen Kontext (Oberarzt, Hausarzt)

## Positionsbasierte und syntaktische Kontextregeln

Was tatsächlich helfen würde:

1. **Positionskontext**: Berufe in Header-Sektionen (vor "Patient:") sind wahrscheinlich Mitarbeiterrollen. Berufe in "Sozialanamnese" oder "Beruf:" Sektionen sind Patientendaten.

2. **Syntaktische Muster**:
   - "Behandelnder X:" → strukturell
   - "arbeitet als X" → Patientenberuf
   - "ist X von Beruf" → Patientenberuf

3. **Sektionsbewusste Verarbeitung**: Dokument erst in Sektionen parsen, unterschiedliche Regeln pro Sektion anwenden.

Hier könnte eine LLM-Schicht Mehrwert bieten. Sie versteht dass "Behandelnder Arzt" ein Label ist und nicht jemandes Job beschreibt.

## Entity-Overlap-Auflösung

Wenn PERSON und EMAIL_ADDRESS überlappen: Welche gewinnt?

- Aktuell: Höchster Score gewinnt (kann Emails an Personennamen verlieren)
- Besser: Spezifischeren Entity-Typ bevorzugen (EMAIL > PERSON Substring)

## Chunking löst Attention-Probleme nicht

Initiale Hypothese: Zweites "58-jähriger" verpasst wegen Chunk-Grenze.

Widerlegt: Mit 200-Zeichen Overlap ist Position 2200 vollständig in Chunk 2 (startet bei 2000). Die Entity sollte erkannt werden.

Tatsächliche Ursache: Attention Competition innerhalb des Chunks, nicht Grenzprobleme.

Mehr Overlap hilft nicht wenn das Problem intra-chunk Attention-Limits sind.

## Was klinische Text-Anonymisierung erfordert

1. **Mehrere Erkennungsmethoden**: Kein einzelner Ansatz findet alles
2. **Entity-Typ-bewusstes Merging**: EMAIL_ADDRESS sollte PERSON Substring schlagen, nicht nach Score konkurrieren
3. **Attention-bewusstes Chunking**: Entity-Dichte berücksichtigen, nicht nur Token-Limits
4. **Validierungspass**: Auf Inkonsistenzen prüfen (gleiches Muster an einer Stelle erkannt, an anderer verpasst)
5. **Quasi-Identifikator-Bewusstsein**: Namen reichen nicht; AGE + OCCUPATION + LOCATION = Re-Identifikation

## Prioritätsprinzip: Lieber False Positives als False Negatives

Bei der Anonymisierung klinischer Daten ist ein verpasster Identifier (False Negative) gefährlicher als eine fälschlich anonymisierte Stelle (False Positive):

- **False Negative**: Personenbezogene Daten bleiben im Text → DSGVO-Verstoß, Re-Identifikationsrisiko
- **False Positive**: Medizinischer Begriff wird unnötig ersetzt → Informationsverlust, aber kein Datenschutzproblem

Deshalb: Im Zweifel anonymisieren. Aber mit Augenmaß, sonst wird der Text unbrauchbar (siehe Over-Anonymization).

Die Kunst liegt darin die Schwelle so zu setzen dass kritische Identifier sicher erkannt werden ohne den Text zu zerstören.

## Wert des Ausprobierens

Viele dieser Probleme wären ohne praktisches Testen nicht aufgefallen:

- Dass "Patient" als Beruf erkannt wird ist nicht vorhersehbar
- Dass Tokenisierung "Jahre alt" zerschneidet zeigt sich erst im Einsatz
- Dass medizinische Fachbegriffe wie "TIMI-3-Fluss" Ortsnamen enthalten erkennt man erst bei der Analyse
- Dass frühere Entities im Chunk spätere unterdrücken war überraschend
- Dass Emails in namenreichen Dokumenten zu PERSON-Fragmenten werden war unerwartet

Die Kombination aus systematischem Testen und Fehleranalyse führt zu konkreten Verbesserungsvorschlägen (Ausschlusslisten, Kontextregeln, Entity-Typ-Priorisierung) die ohne diese Findings nicht entstanden wären.

---

# Teil 3: Erkenntnisse aus der Metrik-Evaluation

**Testaufbau**: Synthetischer deutscher Entlassungsbrief (~770 Wörter) mit 71 annotierten Entitäten. Methoden: Pattern-Recognizer (8 deutsche IDs), GLiNER (Zero-shot NER), LLM (Ministral 8B/14B via Ollama).

Findings aus der Implementierung von Precision/Recall/F2-Metriken mit Ground-Truth-Annotationen.

## Annotation-Qualität schlägt Modell-Tuning

Die größte Verbesserung der Metriken kam nicht durch Modellverbesserungen, sondern durch Korrektur der Annotationen.

| Änderung | Auswirkung auf Pattern-Precision |
|----------|----------------------------------|
| Original-Annotationen | 57% |
| Korrigierte Annotationen | **93%** |

**Ursache**: Die ursprünglichen Annotationen hatten falsche Span-Grenzen. "81377 München" war als DE_POSTAL_CODE annotiert, aber das Modell erkennt korrekt "81377" (PLZ) und "München" (ORT) separat.

**Learning**: Vor Modell-Optimierung immer erst Annotationsqualität prüfen. Schlechte Annotationen führen zu irreführenden Metriken.

## LLMs finden Entitäten einmal, verpassen Wiederholungen

LLMs neigen dazu, jede Entität nur beim ersten Vorkommen zu erkennen.

| Entität | Vorkommen im Text | LLM findet |
|---------|-------------------|------------|
| "München" | 8x | 1x |
| "Hauptstraße 42" | 3x | 1x |
| "05.01.2026" | 2x | 1x |

**Lösung**: Post-Processing-Expansion – nach LLM-Erkennung alle Vorkommen jedes gefundenen Textes im Dokument suchen.

*Auswirkung (gemessen während Entwicklung):*
| Metrik | Vor Expansion | Nach Expansion |
|--------|---------------|----------------|
| Recall | 50.0% | **63.9%** |
| Precision | 97.3% | 97.9% (gehalten) |
| F2 | 55.4% | **68.7%** |

+14 Prozentpunkte Recall ohne Precision-Verlust durch simples String-Matching.

## Prompt-Engineering wirkt auch bei kleinen Modellen

Einfache Prompt-Änderungen verbesserten das 8B-Parameter-Modell deutlich:

**Vorher**:
```
"Extract ALL entities found. Find ALL entities, not just one."
```

**Nachher**:
```
"CRITICAL: Find EVERY occurrence of each entity. If a city name appears 5 times, return 5 separate entries."

"German-specific patterns:
- PERSON: Include 'Herr/Frau [Name]', titles with surnames
- LOCATION: Streets ('Königsallee 77'), hospital wards ('Station 4C')
- OCCUPATION: German compounds like 'Lehrer', 'Ingenieurin'"
```

*Auswirkung (gemessen während Entwicklung):*
| Metrik | Vor Prompt-Verbesserung | Nach Prompt-Verbesserung |
|--------|-------------------------|--------------------------|
| Precision | 85.3% | **97.3%** |
| Recall | 40.3% | 50.0% |

Konkrete Beispiele und explizite Wiederholungs-Anweisung halfen mehr als vage Instruktionen.

## Drei verschiedene Konzepte: Erkennung vs. Annotation vs. Anonymisierung

| Konzept | Frage | Beispiel |
|---------|-------|----------|
| **NER-Fähigkeit** | Was KANN das Modell erkennen? | "Oberarzt Kardiologie" als OCCUPATION |
| **Ground Truth** | Was WOLLEN wir erkennen? | Nur Patienten-Berufe, nicht Arzt-Rollen? |
| **Anonymisierungs-Policy** | Was SOLL ersetzt werden? | "Bankangestellter" ja, "Oberarzt" nein |

**Problem**: "Oberarzt Kardiologie" wurde annotiert weil es erkannt wird – aber es ist keine Patienten-PII.

**Konsequenz**: Ground-Truth sollte die Anonymisierungs-Policy reflektieren, nicht die NER-Fähigkeit. Sonst misst man das Falsche.

## IoU-Threshold beeinflusst Metriken erheblich

Verschiedene Overlap-Schwellwerte für Span-Matching:

| Threshold | TP | FP | FN | Recall |
|-----------|----|----|-----|--------|
| 0.7 (strikt) | 33 | 72 | 28 | 54.1% |
| 0.5 (Standard) | 43 | 62 | 18 | 70.5% |
| 0.3 (lenient) | 46 | 59 | 15 | **75.4%** |

Bei 0.3 werden partielle Matches wie "Hauptstraße 42" vs. "Hauptstraße 42, 80331 München" akzeptiert.

**Empfehlung**: Für klinische Anonymisierung ist 0.3-0.5 sinnvoll – ein partiell erkannter Name ist besser als gar nicht erkannt.

## Was NICHT anonymisiert werden sollte

Nicht jede erkennbare Entität ist Patienten-PII:

| Entität | Anonymisieren? | Begründung |
|---------|----------------|------------|
| Patienten-Name | **Ja** | Direkter Identifier |
| Arzt-Name | **Ja** | Könnte Patient indirekt identifizieren |
| Arzt-Rolle ("Oberarzt") | **Nein** | Kein Patienten-Datum, medizinischer Kontext |
| "Bankangestellter" | **Ja** | Patienten-Beruf, Quasi-Identifier |
| "Universitätsklinikum" | **Ja** | Geografischer Identifier |
| "Klinik" (generisch) | **Nein** | Strukturelles Label |
| Symptome, Diagnosen | **Nein** | Medizinischer Inhalt |

**Implikation für Prompts**: Dem LLM zu sagen "finde keine Arzt-Berufe" könnte ein 8B-Modell verwirren. Besser: Alle erkennen, dann per Policy filtern.

## PLZ + Stadtname: Semantik vs. Erkennung

"81377 München" ist semantisch EINE Ortsangabe, aber:
- Pattern erkennt "81377" als DE_POSTAL_CODE
- LLM erkennt "München" als LOCATION

**Annotations-Dilemma**:
- Option A: Als Compound "81377 München" annotieren → Modelle werden bestraft
- Option B: Getrennt annotieren → Passt zu Modell-Output, aber künstlich
- Option C: Beides erlauben → Komplexere Evaluation

**Aktueller Ansatz**: Option B (getrennt), mit dem Bewusstsein dass dies die Ergebnisse etwas "schönt".

## Methodenstärken gezielt nutzen

Jede Methode hat klare Stärken:

| Methode | Stärke | Schwäche |
|---------|--------|----------|
| Pattern | ~93% Precision bei DE-IDs, deterministisch | Kann keine Namen/Orte |
| GLiNER | Breiter Recall (~66%) | ~45% Precision, viele FPs |
| LLM | Hohe Precision (95-100%) | Moderater Recall (~75%), konservativ |

**Optimale Kombination** (implementiert):
- Pattern für DE_KVNR, DE_LANR, DE_BSNR, etc. (vertrauen)
- LLM für PERSON, EMAIL, PHONE (hohe Precision)
- GLiNER ausgeschlossen (viele False Positives ohne Mehrwert)

**Aktuelle Ergebnisse (Pattern + LLM):**
| Modell | Precision | Recall | F2-Score |
|--------|-----------|--------|----------|
| 8b | **95%** | **78%** | **80%** |
| 14b | **100%** | **76%** | **80%** |

Domänenspezifisch (14b):
- Pattern-Domain (DE_* IDs): P=100% R=79% F2=82%
- LLM-Domain (Namen, Orte, etc.): P=100% R=75% F2=79%

## Zusammenfassung der Verbesserungen

| Maßnahme | Auswirkung |
|----------|------------|
| Annotations-Korrektur | Baseline korrekt |
| Occurrence-Expansion | LLM findet alle Vorkommen erkannter Entitäten |
| GLiNER-Entfernung | Precision drastisch verbessert (weniger False Positives) |
| **Gesamt (Regex+LLM 8b)** | **P=95% R=78% F2=80%** |
| **Gesamt (Regex+LLM 14b)** | **P=100% R=76% F2=80%** |

Die größte Verbesserung kam durch das Entfernen von GLiNER: Die vielen False Positives zerstörten die Precision der Kombination.

## Domänenspezifische Evaluation ist entscheidend

Die Gesamtmetriken sind irreführend. Jede Methode hat ihren Zielbereich:

| Methode | Pattern-Domain (DE_* IDs) | LLM-Domain (Namen, Orte, etc.) |
|---------|---------------------------|--------------------------------|
| **Pattern** | P=93% R=100% **F2≈99%** | 0% (nicht zuständig) |
| **GLiNER** | P=100% R=50% F2=56% | P=41% R=70% F2=61% |
| **LLM (8b)** | 0% (nicht zuständig) | P=94% R=75% **F2=79%** |
| **Regex+LLM (8b)** | P=100% R=86% **F2=88%** | P=94% R=75% **F2=79%** |

**Erkenntnis**: Das LLM erreicht **79% F2** auf seinem Zielbereich (Namen, Orte, etc.). Die Pattern-Recognizer erreichen **88% F2** auf deutschen IDs. Jede Methode sollte nur auf ihrer Zieldomäne bewertet werden.

**Konsequenz für Evaluation**: Methoden sollten nur auf ihrer Zieldomäne gemessen werden. Pattern-Recognizer für strukturierte IDs, LLM für kontextabhängige Entitäten.

## GLiNER bringt keinen Mehrwert in der Kombination

GLiNER ist auf beiden Domänen schlechter als die Spezialmethode:
- Pattern-Domain: GLiNER F2=56% vs Pattern F2≈99%
- LLM-Domain: GLiNER F2=61% vs LLM F2≈79%

Die vielen False Positives von GLiNER würden die Precision der Kombination drastisch herunterziehen.

**Umgesetzt**: GLiNER aus der produktiven Kombination entfernt. Als Vergleich im Demo-Report weiterhin sichtbar, aber nicht in das kombinierte Ergebnis einbezogen. Die optimale Kombination ist **Regex+LLM ohne GLiNER**.

## Naive Kombination erzeugt Konflikte

Wenn Pattern "81377" als DE_POSTAL_CODE (Score 0.85) erkennt und LLM "81377 München" als LOCATION (Score 0.85), gewinnt bei Überlappung die größere Span. Das führt zu:
- Verlust der präzisen PLZ-Erkennung
- Falsche Entity-Typ-Zuordnung

Eine intelligente Aggregation müsste:
1. Pattern-Erkennern bei deutschen IDs Vorrang geben
2. LLM-Erkennungen bei Namen/Orten bevorzugen
3. Bei Überlappung die domänenspezifisch zuverlässigere Methode wählen

Dies ist einer der Gründe, warum ein agentenbasierter Ansatz sinnvoll ist.

---

# Teil 4: Warum ein agentenbasierter Ansatz?

Die vorherigen Teile dokumentieren die Grenzen der naiven Single-Pass-Pipeline: 76-78% Recall bedeutet, dass bei 100 PII-Vorkommen 22-24 im Text verbleiben. Für klinische Anwendungen ist das ein erhebliches Risiko.

## Das Problem ist strukturell, nicht komponentenbasiert

Die Findings zeigen: Pattern-Recognizer erreichen auf ihrem Zielbereich nahezu perfekte Ergebnisse. Das LLM performt gut auf kontextabhängigen Entitäten. Die Komponenten sind nicht das Problem.

Das Problem ist die **naive Kombination in einem einzigen Durchlauf**:
- Keine Iteration nach dem ersten Ergebnis
- Keine Rückkopplung bei inkonsistenten Erkennungen
- Keine kontextabhängige Entscheidungslogik

## Was ein Agent anders machen kann

Ein agentenbasierter Ansatz verwendet **dieselben Werkzeuge**, orchestriert sie aber intelligent:

**Iteration statt Single-Pass**: Nach dem ersten Durchlauf prüfen: Wurden alle Vorkommen eines Namens markiert? Gibt es Widersprüche zwischen Pattern- und LLM-Ergebnissen? Hypothesen testen, Ergebnisse korrigieren.

**Fokussierte Phasen statt Prompt-Überladung**: Statt alle Anforderungen in einen Prompt zu packen, sequentielle Schritte mit je einer klaren Aufgabe.

**Selbstverbesserung**: Systematische Analyse, wo die Pipeline versagt. Gezielte Anpassungen basierend auf empirischen Schwächen.

## Vision

Von 76-78% Recall auf deutlich höhere Werte – durch iterative Verfeinerung statt besserer Einzelkomponenten.

> Ein einzelner naiver Durchlauf wird die verbleibenden 22-24% nicht schließen. Aber iterative agentenbasierte Schleifen ermöglichen **graduelle Verbesserung**: Jeder Analyse-Zyklus kann identifizierte Lücken schließen.

---

Die Implementierung dieses Ansatzes erfolgt im [PIIgent-Projekt](https://github.com/shokrydev/piigent).

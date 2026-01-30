# German PII Recognizers

This directory contains pattern-based recognizers for German Personally Identifiable Information (PII) in compliance with DSGVO/GDPR requirements.

**Special Focus on Clinical/Healthcare Data**: Includes recognizers for German healthcare identifiers required for anonymizing electronic health records, prescriptions, discharge letters, and all clinical texts.

## Recognizer Overview

### Clinical/Healthcare Identifiers

| Recognizer | Entity Type | Pattern | Context Words | Validation |
|------------|-------------|---------|---------------|------------|
| **DeKvnrRecognizer** | `DE_KVNR` | 1 letter + 9 digits | kvnr, krankenversichertennummer, versichertennummer | Modified Luhn checksum |
| **DeTelematikIdRecognizer** | `DE_TELEMATIK_ID` | (1\|10)-[A-Z0-9.]{5,125} | telematik-id, gesundheits-id, ehba | Format only |
| **DeLanrRecognizer** | `DE_LANR` | 9 digits | lanr, arztnummer, arzt-nr | Format only |
| **DeBsnrRecognizer** | `DE_BSNR` | 9 digits | bsnr, betriebsstättennummer, praxisnummer | Regional code validation |

### Personal Documents & Government IDs

| Recognizer | Entity Type | Pattern | Context Words | Validation |
|------------|-------------|---------|---------------|------------|
| **DePersonalIdRecognizer** | `DE_PERSONAL_ID` | 9 chars (new) / 10 chars (MRZ) / 10 digits (old) | personalausweis, ausweisnummer, perso | Checksum (7,3,1 weighted) |
| **DePassportRecognizer** | `DE_PASSPORT` | 9 chars starting C,F,G,H,J,K | reisepass, passnummer | Checksum (7,3,1 weighted) |
| **DeDriverLicenseRecognizer** | `DE_DRIVER_LICENSE` | 10-15 alphanumeric | führerschein, fahrerlaubnis | Format only |
| **DeTaxIdRecognizer** | `DE_TAX_ID` | 11 digits | steuer-id, steuerid, idnr, tin | ISO 7064 MOD 11,10 |
| **DeSocialSecurityRecognizer** | `DE_SOCIAL_SECURITY` | 12 chars (BB DDMMYY L SSP) | svnr, rvnr, rentenversicherungsnummer | Area code + date + checksum |

### Business & Infrastructure

| Recognizer | Entity Type | Pattern | Context Words | Validation |
|------------|-------------|---------|---------------|------------|
| **DeCommercialRegisterRecognizer** | `DE_COMMERCIAL_REGISTER` | HRA/HRB + 1-7 digits | handelsregister, hrb, hra | Format only |
| **DePostalCodeRecognizer** | `DE_POSTAL_CODE` | 5 digits (01001-99998) | plz, postleitzahl | Range check |
| **DeLicensePlateRecognizer** | `DE_LICENSE_PLATE` | [1-3 letters district][1-2 letters][1-4 numbers][E/H] | kennzeichen, kfz | Format + E/H suffix |
| **DeVatCodeRecognizer** | `DE_VAT_CODE` | DE + 9 digits (11 chars total) | ust-id, umsatzsteuerid | Format only |

## Pattern Specifications

### Clinical/Healthcare Identifiers

#### KVNR (Krankenversichertennummer)
- **Format:** 10 characters total
  - Position 1: One uppercase letter (A-Z, no umlauts)
  - Positions 2-9: Eight random digits
  - Position 10: Check digit
- **Example:** `A123456789`, `K987654321`
- **Checksum:** Modified Luhn algorithm with weights 1-2-1-2-1-2-1-2-1-2
  - Letter converted to 2-digit number (A=01, B=02, ..., Z=26)
  - Creates 11-digit number for checksum calculation
- **Prevalence:** Appears in **every** German clinical document
- **Legal Basis:** §290 SGB V
- **Issuing Authority:** ITSG (Informationstechnische Servicestelle der GKV)
- **Note:** Primary patient identifier in German healthcare system

#### LANR (Lebenslange Arztnummer)
- **Format:** 9 digits total
  - Digits 1-6: Unique physician identifier (lifetime)
  - Digit 7: Check digit (Prüfziffer)
  - Digits 8-9: Specialty group code (Fachgruppenschlüssel)
- **Example:** `123456789`
- **Prevalence:** Prescriptions, discharge letters, billing documents
- **Legal Basis:** §75 Abs. 7 SGB V
- **Issuing Authority:** Kassenärztliche Bundesvereinigung (KBV)
- **Note:** Identifies treating physician; quasi-PII as it can reveal specialist treating condition
- **Multiple Numbers:** Physicians can have multiple LANRs if practicing in multiple specialties (first 7 digits remain the same)

#### BSNR (Betriebsstättennummer)
- **Format:** 9 digits total
  - Digits 1-2: KV state/regional association code (01-99)
  - Digits 3-7: Facility identifier (assigned by KV)
  - Digits 8-9: Additional digits (often "00" for older BSNRs)
- **Example:** `987654321`
- **Prevalence:** Prescriptions, discharge letters, billing documents
- **Legal Basis:** §75 Abs. 7 SGB V
- **Issuing Authority:** Kassenärztliche Vereinigungen (KV)
- **Note:** Identifies treatment facility/practice location; quasi-PII for geo-locating patient treatment
- **Historical Note:** Corresponds to pre-2008 7-digit KV-Abrechnungsnummer + "00"

#### Telematik-ID (Including Gesundheits-ID)
- **Format:** `[Prefix][Separator][Continuation]`
  - **Prefix 1-:** Healthcare professionals (eHBA - elektronischer Heilberufsausweis)
    - Doctors, nurses, therapists, pharmacists
    - LIFELONG identifier, independent of workplace
  - **Prefix 10-:** Patients (Gesundheits-ID)
    - Digital patient identifier in German healthcare telematics infrastructure
  - **Prefix 5-2-:** Hospital institutions (SMC-B cards with IK-Nummer)
  - **Prefix 9-:** Organization cards issued by gematik
  - **Prefix 11-:** Health craft workers (format: `11-{card type}.{chamber-ID}.{individual ID}`)
- **Character Set:** `0-9`, `A-Z`, `-`, `.`
- **Max Length:** 128 characters
- **Examples:**
  - Healthcare professional: `1-10000100001111`
  - Patient (Gesundheits-ID): `10-ABC123XYZ789`
  - Hospital: `5-2-123456789012`
- **Prevalence:** Electronic health records, e-prescriptions, clinical documentation
- **Legal Basis:** gemSpec_PKI (gematik specification)
- **Issuing Authority:** Gematik, Chambers (Kammern), KV/KZV
- **Note:** The Gesundheits-ID (prefix `10-`) is a digital patient identifier used in German healthcare IT infrastructure

### Personal Documents & Government IDs

#### Personal ID (Personalausweisnummer)
- **New format (since Nov 2010):** 9 alphanumeric characters (front of card)
- **MRZ format:** 10 alphanumeric characters (9 document number + 1 check digit from back of card)
- **Old format (before Nov 2010):** 10 digits
- **Character set:** C, F, G, H, J, K, L, M, N, P, R, T, V, W, X, Y, Z, 0-9
- **Excluded:** Vowels (A, E, I, O, U) and confusable letters (B, D, Q, S)
- **Checksum:** Weighted sum algorithm (7, 3, 1)
- **Source:** [Wikipedia - German identity card](https://en.wikipedia.org/wiki/German_identity_card)

### Passport (Reisepass)
- **Format:** 9 alphanumeric characters
- **First character:** Must be C, F, G, H, J, or K
- **Character set:** Excludes vowels (A, E, I, O, U) and confusable letters (B, D, Q, S)
- **Checksum:** Weighted sum algorithm (7, 3, 1)

### Driver License (Führerschein)
- **Format:** Varies by issuing authority and period
- **EU card format (since 2013):** Typically 11 alphanumeric characters
- **Structure:** [Authority code][Serial number][Check digit]

### Tax ID (Steueridentifikationsnummer)
- **Format:** 11 digits
- **Rules:** First digit cannot be 0; one digit appears 2-3 times; at least one digit (0-9) doesn't appear
- **Checksum:** ISO 7064 MOD 11,10 algorithm

### Social Security (Sozialversicherungsnummer)
- **Format:** 12 characters (BBDDMMYYASSP)
  - BB: Area number (Bereichsnummer)
  - DDMMYY: Birth date
  - A: First letter of birth surname
  - SS: Serial number
  - P: Check digit

### Commercial Register (Handelsregisternummer)
- **Format:** HRA or HRB + 1-7 digit number
- **HRA:** Partnerships (OHG, KG, e.K.)
- **HRB:** Corporations (GmbH, AG, UG)

### Postal Code (Postleitzahl)
- **Format:** 5 digits (01001-99998)
- **Structure:** First 2 digits = region; last 3 digits = district

### License Plate (Kfz-Kennzeichen)
- **Format:** [1-3 letter district code][1-2 letters][1-4 numbers][optional E/H suffix]
- **District codes:** May include umlauts (Ä, Ö, Ü), e.g., TÜ for Tübingen
- **Suffix:** E = electric/hybrid vehicle, H = historic vehicle (Oldtimer 30+ years)
- **Examples:** B-AB 1234, M-XY 123E (electric), HH-OL 99H (historic)
- **Max total:** 8 characters (excluding separators and suffix)

### VAT Number (USt-IdNr)
- **Format:** DE + 9 digits = 11 characters total
- **Example:** DE123456789

## Sources

Each recognizer includes an authoritative pattern source reference in its code comments. See the individual recognizer files for official documentation links.

## Usage

### Basic Usage

```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import (
    DeTaxIdRecognizer,
    DePassportRecognizer,
    # ... other German recognizers
)

# Add to analyzer
analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(DeTaxIdRecognizer())

# Analyze text
results = analyzer.analyze(
    text="Meine Steuer-ID ist 12345678903",
    language="de",
    entities=["DE_TAX_ID"]
)
```

### Clinical Text Anonymization

```python
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import (
    DeKvnrRecognizer,
    DeLanrRecognizer,
    DeBsnrRecognizer,
)

# Setup analyzer for clinical text
analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(DeKvnrRecognizer())
analyzer.registry.add_recognizer(DeLanrRecognizer())
analyzer.registry.add_recognizer(DeBsnrRecognizer())

# Analyze clinical discharge letter
clinical_text = """
Entlassungsbrief

Patient: Max Mustermann
KVNR: A123456789
Behandelnder Arzt: Dr. Schmidt (LANR: 987654321)
Praxis: Hausarztpraxis München (BSNR: 123456700)

Diagnose: Hypertonie (ICD-10: I10)
Medikation: Ramipril 5mg
"""

results = analyzer.analyze(
    text=clinical_text,
    language="de",
    entities=["DE_KVNR", "DE_LANR", "DE_BSNR", "PERSON"]
)

# Results will identify:
# - KVNR: A123456789
# - LANR: 987654321
# - BSNR: 123456700
# - Person names (via NER models)
```

## Validation Logic

Each recognizer implements a `validate_result()` method that returns:
- `True`: Pattern passes checksum/validation (high confidence)
- `False`: Pattern fails validation (rejected)
- `None`: No definitive validation available; use pattern score with context enhancement

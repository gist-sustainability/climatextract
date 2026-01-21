# Prompts

climatextract uses carefully structured prompts to instruct the LLM on extracting emissions data. This page explains the prompt design.

---

## Prompt Structure

Each prompt consists of three parts:

1. **Role** – Defines the LLM's persona
2. **Task** – Specifies what to extract
3. **Specifications** – Rules and constraints

```mermaid
flowchart LR
    R[Role] --> P[Full Prompt]
    T[Task + Definitions] --> P
    S[Specifications] --> P
    C[Page Content] --> P
    P --> LLM[LLM]
```

---

## Role Definition

The LLM is instructed to act as a climate analyst:

> "You are a climate analyst tasked with extracting specific absolute numerical data from corporate reports. Your objective is to extract only the absolute values for the following Key Performance Indicators (KPIs) related to CO2 emissions across the entire company."

---

## KPI Definitions

The prompt provides clear definitions for each scope:

| Scope | Definition |
|-------|------------|
| **Scope 1** | Direct GHG emissions from sources owned or controlled by the organization |
| **Scope 2** | Indirect GHG emissions from purchased electricity, steam, heating, and cooling |
| **Scope 3** | Indirect GHG emissions from the organization's value chain (upstream and downstream) |

---

## Extraction Specifications

The prompt includes rules to ensure data quality:

- Only extract values for the **whole company** (not divisions/subsidiaries)
- Exclude percentage changes or relative values
- Exclude targets or forecasts
- Extract **separate** Scope 1, 2, 3 values (not combined totals)
- Return `null` if data is not available

---

## Output Format

The LLM is instructed to return JSON matching this schema:

```json
{
  "KPI_Entries": [
    {
      "year": 2023,
      "scope": "1",
      "value": 55000.0,
      "unit": "tCO2e"
    }
  ]
}
```

This is validated using Pydantic models to ensure data quality.

---

## Prompt Types

climatextract supports two prompt types:

### Default Prompt

Standard extraction with regex-based parsing. Good for simpler use cases.

```toml
[extraction]
prompt_type = "default"
```

### Custom GAIA Prompt

Advanced prompt with Pydantic-based structured output parsing. Recommended for production use.

```toml
[extraction]
prompt_type = "custom_gaia"
```

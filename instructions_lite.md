# Project Brief: Fantasy Football Projection Aggregator & Normalizer (MVP)

**Date:** May 7, 2025

## 1. Project Goal (MVP)

To develop a Python-based tool that ingests fantasy football player projection data from multiple CSV sources (and previous year's actual stats), then cleans, standardizes, normalizes, scales, and merges this data into a single, unified master DataFrame. This DataFrame will provide a consistent and comparable set of player projections, keyed by unique player identifiers, suitable for further analysis.

## 2. Core Problem Solved

This project addresses the challenge of aggregating and standardizing messy, disparate fantasy football player projection data. Sources often vary significantly in:
* Player naming conventions (e.g., "Ken Walker" vs. "Kenneth Walker III").
* Team and position abbreviations.
* Column names and data availability.
* Overall scale of point projections (some experts are inherently more optimistic/pessimistic).

## 3. Key Inputs

* **N CSV files:** Current season player projections (PPR points focused) from N distinct providers, placed in a designated input directory.
* **1 CSV file:** Previous (last full) season's actual player statistics and PPR fantasy points, serving as a benchmark.
* **Configuration Files (YAML/JSON):** For defining column mappings, team/position standardization rules, player normalization parameters (e.g., fuzzy match thresholds), and projection scaling settings.
* **Manual Override Files (CSV):** User-maintained files to enforce specific player name matches (`manual_overrides.csv`) and assist normalization, allowing iterative refinement.

## 4. Core Processing Pipeline (High-Level Stages for MVP)

The project is broken down into distinct, sequential processing stages, each typically corresponding to a Python script:

1.  **Data Loading (`data_loader.py`):**
    * Reads N projection CSVs and 1 historical CSV for the target season.
    * Tags data with its source expert identifier.

2.  **Data Cleaning & Standardization (`data_cleaner.py`):**
    * Standardizes column names (e.g., to `player_name`, `team`, `position`, `ppr_projection`).
    * Normalizes team and position abbreviations using configurable mappings.
    * Performs basic data type conversions and cleaning (whitespace, ensure numeric projections).

3.  **Player Name Normalization (`player_normalizer.py`):**
    * Assigns a unique, system-generated `canonical_player_id` to each distinct player across all data sources (projections and historical).
    * Employs a multi-pass strategy:
        1.  Applies `manual_overrides.csv` (highest precedence).
        2.  Uses a generated `short_key` (team + position + first N chars of cleaned name) for initial, high-confidence grouping.
        3.  Performs fuzzy string matching (e.g., using `thefuzz`) on names within these groups and against a growing `master_player_roster.csv`, strongly considering exact team/position matches to confirm similarity.
    * Outputs ambiguous/low-confidence matches to a `needs_review.csv` for manual intervention and iterative improvement of overrides.

4.  **Projection Scaling/Benchmarking (`projection_scaler.py`):**
    * Adjusts each expert's raw PPR projections to a common reference scale to mitigate source-specific optimism/pessimism.
    * Defines "benchmark player sets" (e.g., Top 20 QBs, Top 40 RBs from the previous season's actual stats).
    * Calculates position-specific scaling factors for each expert by comparing their projections for these benchmark sets against either:
        * The historical actual point totals for these sets.
        * The average of all current-year expert projections for these sets (configurable).
    * Applies these factors to create `scaled_projection_points_ppr`.

5.  **Data Merging (`data_merger.py`):**
    * Consolidates all processed (cleaned, normalized, scaled) projections into a single master Pandas DataFrame.
    * Uses the `canonical_player_id` as the primary merge key.
    * The DataFrame is indexed by a player display name, with columns for `player_id` (canonical), `team`, `position`, and `proj_scaled_{EXPERT_ID}` for each expert (optionally, original projections are also included).

## 5. Key Output (MVP)

* **Primary Deliverable:** A single Pandas DataFrame (e.g., `merged_player_projections.parquet` or `.csv`) saved locally. This DataFrame contains uniquely identified players with their standardized metadata and a comparable set of scaled (and optionally original) PPR projections from all expert sources.
* **Supporting Files:** Updated `master_player_roster.csv` and potentially `needs_review.csv`.

## 6. Key Technologies/Libraries (Anticipated)

* **Language:** Python 3.x
* **Core Libraries:** Pandas (data manipulation), `thefuzz` (fuzzy string matching).
* **Configuration:** YAML or JSON for external configuration files.

## 7. Guiding Principles/Approach

* **Modularity:** Each core stage is a distinct component.
* **Configuration-Driven:** Externalize mappings, rules, and parameters for flexibility.
* **Iterative Refinement:** Design for continuous improvement, especially for player name normalization, through manual review and override mechanisms.
* **Prioritize Accuracy:** Emphasize correct data linkages, particularly for players, using high-confidence rules and manual checks where ambiguity exists.

## 8. Future Scope (Beyond Current MVP)

While the current focus is on producing the merged/scaled projection DataFrame, future enhancements could include:
* Integrating league format settings to calculate player "value" scores (e.g., VORP).
* Developing dynamic value updates as players are selected in a mock/live draft.
* Building a draft simulation tool.
* Automating data procurement.
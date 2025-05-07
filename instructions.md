# Fantasy Football Projection Aggregator & Draft Tool - instructions.md

## 1. Project Overview

### 1.1. Goal
To create a Python-based tool that:
1.  Aggregates expert fantasy football projections from multiple CSV sources and historical performance data for a given season.
2.  Cleans, standardizes, and merges this data into a consensus player projection set.
3.  Calculates player "value" based on configurable league settings and positional scarcity.
4.  Dynamically updates player values during a simulated or live draft.
5.  Simulates drafts to help identify optimal drafting strategies.
(Target User: Primarily for self-use, CLI-based.)

### 1.2. Core Stages & Corresponding Python Files (Proposed)
* **Configuration:** (`config.py` or `settings.yaml`) - For storing paths, season year, constants, league settings (can be overridden), list of desired optional metadata.
* **Main Orchestrator:** (`main.py` or `app.py`) - To coordinate the workflow based on the active season year.
* **Utilities:** (`utils.py`) - Common helper functions (e.g., logging setup, file I/O wrappers).

* **Data Procurement & Loading:** (`data_loader.py`)
    * Loads N projection CSVs from `data/input/{season_year}/projections/`.
    * Loads 1 historical actuals CSV from `data/input/{season_year}/historical/`.
* **Data Cleaning & Standardization:** (`data_cleaner.py`)
    * Handles initial cleaning (e.g., whitespace, data types).
    * Standardizes column names from various sources to internal conventions (e.g., 'Player Name', 'Team', 'Position', 'Projected_PPR').
    * Standardizes team abbreviations.
    * Standardizes position abbreviations.
* **Player Name Normalization:** (`player_normalizer.py`)
    * Sophisticated matching of player names across different sources (e.g., "Kenneth Walker", "Kenneth Walker III").
* **Projection Scaling/Benchmarking:** (`projection_scaler.py`)
    * Uses historical data to normalize the scale of projections from different experts before merging.
* **Data Merging:** (`data_merger.py`)
    * Combines N cleaned and scaled expert projections into a single master projection DataFrame.
    * Carries through essential (Name, Team, Pos, Proj_PPR_ExpertX) and defined optional metadata.
* **League Configuration Management:** (`league_manager.py` or part of `config.py`)
    * Defines and allows selection/editing of league settings (number of teams, roster spots per position, flex logic, scoring rules if needed beyond PPR).
* **Value Calculation:** (`value_calculator.py`)
    * Determines player values (e.g., Value Over Replacement Player - VORP, or similar) based on consensus projections and active league settings.
* **Draft Engine/State Management:** (`draft_engine.py`)
    * Manages the state of a draft: available players, taken players (and by which team/pick), current pick.
* **Dynamic Value Updater:** (`dynamic_updater.py`)
    * Recalculates player values after each pick based on remaining players and team needs.
* **Draft Simulator:** (`draft_simulator.py`)
    * Runs simulated drafts using different strategies or ADP logic.
* **Output Generation:** (`output_generator.py`)
    * Saves processed dataframes (e.g., merged projections, value lists).
    * Generates reports or summaries.

### 1.3. MVP Definition
* **Part 1 (Core):** Successfully ingest, clean (basic), standardize (column names, team/pos abbreviations), normalize player names, and merge N expert projections into a single DataFrame. This DataFrame will have player names as the index (or a unique player ID) and columns for each expert's projection (e.g., `proj_expert1`, `proj_expert2`), along with standardized `Team` and `Position`.
* **Part 2 (Value Add):** Implement the projection scaling using historical data. Implement value calculation based on a defined league format.

### 1.4. Future Enhancements
* Automated download/acquisition of CSVs.
* Handling raw statistical projections (converting to fantasy points).
* Converting ranks to pseudo-projections.
* More sophisticated UI (web app, GUI).
* Advanced draft simulation parameters and analysis.
* Integration with live draft platforms (highly advanced).

## 2. Project Setup

### 2.1. Directory Structure

Okay, this is a fantastic start! I appreciate the structured approach and the immediate dive into clarifying questions. Let's address your high-level questions first, then tackle Data Procurement.

High-Level Clarifications:

Project Goal (Concise): Your summary is spot on: "To create a tool that aggregates expert fantasy football projections, calculates player values dynamically during a draft, and simulates draft outcomes to identify optimal strategies." That captures it perfectly.
Target User: Primarily for myself. I might share the methodology or resulting dataframes with friends, but the tool itself is for my personal use. So, command-line interface (CLI) is perfectly fine; no need for a fancy UI at this stage.
Definition of "Done" (MVP): The "BIG MILESTONE" I mentioned – a dataframe with index "name" and columns: ['proj_expert1', 'proj_expert2', ... , 'proj_expertN'] – is indeed the core of the MVP. If I can reliably get to this merged and reasonably cleaned dataset, I'd consider that a huge win and the foundation for everything else. The value calculation would be MVP part 2, and dynamic updates/simulation would be subsequent enhancements.
Longevity/Maintenance: I plan to use this for multiple seasons. This means the design should accommodate:
Annual updates of projection and historical data.
Potential changes in data sources or their CSV formats over time.
The ability to easily switch between analysis for different seasons (e.g., "2025_season_data", "2026_season_data").
Proposed File Structure:
The proposed file structure looks like a very logical breakdown. I especially like the separation of concerns, like player_normalizer.py and projection_scaler.py. This modularity will be key. config.py or a YAML file for settings is also a great idea.

Now, let's dive into your Specific Questions for Data Procurement:

Number of Sources (N):

Initial N: For the initial version, let's plan for N being between 3 and 7 sources. The system should be built to handle a variable N within a reasonable range without code changes (just new files dropped in).
Consistency of Sources: I anticipate sources might change year-to-year. Some mainstays will likely remain, but I might add or remove one or two each season. So, the system shouldn't be hardcoded to specific expert names derived from filenames, for example.
File Location: The N CSV files (and the 1 historical CSV) will be supplied by me and placed in a designated local directory before running the main script. Let's assume a structure like data/input/{season_year}/projections/ for the N files and data/input/{season_year}/historical/ for the historical stats. Filenames within projections/ could be anything, e.g., expertA_projections.csv, FantasySharks_Week1_PPR.csv, CBS_Sports_Projections_Full_Season.csv.
CSV File Structure (Projections/Rankings):

Column Headers (Examples - highly variable):
Source 1 (expertA.csv): PLAYER NAME, TEAM, POS, PROJ. PTS
Source 2 (expertB.csv): Rank, Name, Team, Position, PPR_Projection
Source 3 (expertC.csv): Full Name, NFL Team, Depth Chart Position, FPts
Source 4 (expertD.csv): Player, PROJ_PPR (might not have team/pos in every file, but most will)
One Expert Per File: Yes, assume one expert/provider per CSV file.
Other Metadata: Files might contain other metadata (age, bye week, injury status, strength of schedule, etc.). For the MVP (merged projections), we primarily care about Player Name, Team, Position, and Projected PPR Points. However, if other useful metadata is consistently available (like Bye Week), we might want to carry it through. Let's make a note to consider this during the cleaning/merging phase – perhaps a configurable list of "desired optional metadata columns."
Projection Format: Primarily PPR point projections. If a source provides raw stats (passing yards, TDs), for the MVP, I will manually convert these to PPR points or find another source. The tool should expect a single "Projected PPR Points" column per expert. We can flag handling raw stats as a future enhancement.
Rankings: For now, let's assume all N files are projections (points). If a source only provides ranks, I will likely skip that source for the MVP or manually convert its ranks to pseudo-projections outside of this tool. So, the tool should focus on finding a numerical projection column. We can consider rank-to-projection conversion as a future feature.
CSV File Structure (Previous Year's Stats):

Expected Stats: Player Name, Team (actual team for that previous year), Pos (actual position), Games Played, Passing Yards, Passing TDs, Interceptions, Rushing Attempts, Rushing Yards, Rushing TDs, Receptions, Receiving Yards, Receiving TDs, Fumbles Lost, Fantasy Points PPR. (Standard stuff).
Benchmarking Process: This is key. The idea is that different projection sources might have different aggregate point totals. For example, Expert A might project 3500 total PPR points for the top 200 players, while Expert B projects 3800. This doesn't mean Expert B is "better" or "more optimistic" in a way that helps us; it's the relative values that matter. The previous year's stats will be used to establish a "benchmark total" for, say, the top N players at each position (e.g., top 30 QBs, top 50 RBs, etc., based on actual previous year PPR scores). Then, for each projection source for the current year, we'd look at their projected totals for the same top N players at each position. We can then scale each expert's projections so that their projected total for these benchmark player sets aligns more closely with the historical benchmark (or an average of all experts' projected totals for that set). This is to ensure that when we average or combine projections, we're not overly swayed by one source that is globally much higher or lower than others. The goal is to normalize the scale of projections before combining. For example, if historical Top 20 RBs scored X points, and Expert A has their Top 20 RBs projected at 1.1X, while Expert B has them at 0.9X, we might scale A down and B up. This is a complex area we'll need to define precisely in the projection_scaler.py section. For now, just knowing this historical file is the basis for this scaling is enough for data procurement.
Data Acquisition Process:

Manual Acquisition: For the MVP, I will download/acquire these CSVs manually and place them in the specified directories.
Automation: Automating downloads is a definite future enhancement, but out of scope for the MVP. The project should assume the files are present.
Data Volume & Performance:

Player Count: Each projection file will likely contain 200-500 players. With 3-7 sources, the total number of rows to process initially (before merging) might be a few thousand.
Performance: Standard Pandas operations should be fine. No major performance concerns anticipated for the MVP with this volume.
Initial Thoughts on File Organization:

Yes, your suggestion is perfect:
data/input/{season_year}/projections/ (for the N expert CSVs)
data/input/{season_year}/historical/ (for the 1 historical actuals CSV)
We'll also need output directories, perhaps:
data/processed/{season_year}/ (for intermediate cleaned files, the merged master projection set)
data/output/{season_year}/ (for final reports, draft simulation results)
The {season_year} placeholder (e.g., 2025) is important for the longevity aspect. The tool should be configurable to run for a specific season.
This clarifies the Data Procurement stage significantly. I'm ready for your next set of questions, perhaps focusing on Data Cleaning and Preprocessing or Player Name Normalization, given how critical and messy that will be. Or, if you see any gaps in my answers above, let's address those first!This is excellent! Your detailed answers for Data Procurement give us a very solid foundation. I agree, the file organization with {season_year} is crucial for multi-season use.

Before we move to Data Cleaning, let's quickly summarize and slightly refine what we have for instructions.md based on your input.

Markdown

# Fantasy Football Projection Aggregator & Draft Tool - instructions.md

## 1. Project Overview

### 1.1. Goal
To create a Python-based tool that:
1.  Aggregates expert fantasy football projections from multiple CSV sources and historical performance data for a given season.
2.  Cleans, standardizes, and merges this data into a consensus player projection set.
3.  Calculates player "value" based on configurable league settings and positional scarcity.
4.  Dynamically updates player values during a simulated or live draft.
5.  Simulates drafts to help identify optimal drafting strategies.
(Target User: Primarily for self-use, CLI-based.)

### 1.2. Core Stages & Corresponding Python Files (Proposed)
* **Configuration:** (`config.py` or `settings.yaml`) - For storing paths, season year, constants, league settings (can be overridden), list of desired optional metadata.
* **Main Orchestrator:** (`main.py` or `app.py`) - To coordinate the workflow based on the active season year.
* **Utilities:** (`utils.py`) - Common helper functions (e.g., logging setup, file I/O wrappers).

* **Data Procurement & Loading:** (`data_loader.py`)
    * Loads N projection CSVs from `data/input/{season_year}/projections/`.
    * Loads 1 historical actuals CSV from `data/input/{season_year}/historical/`.
* **Data Cleaning & Standardization:** (`data_cleaner.py`)
    * Handles initial cleaning (e.g., whitespace, data types).
    * Standardizes column names from various sources to internal conventions (e.g., 'Player Name', 'Team', 'Position', 'Projected_PPR').
    * Standardizes team abbreviations.
    * Standardizes position abbreviations.
* **Player Name Normalization:** (`player_normalizer.py`)
    * Sophisticated matching of player names across different sources (e.g., "Kenneth Walker", "Kenneth Walker III").
* **Projection Scaling/Benchmarking:** (`projection_scaler.py`)
    * Uses historical data to normalize the scale of projections from different experts before merging.
* **Data Merging:** (`data_merger.py`)
    * Combines N cleaned and scaled expert projections into a single master projection DataFrame.
    * Carries through essential (Name, Team, Pos, Proj_PPR_ExpertX) and defined optional metadata.
* **League Configuration Management:** (`league_manager.py` or part of `config.py`)
    * Defines and allows selection/editing of league settings (number of teams, roster spots per position, flex logic, scoring rules if needed beyond PPR).
* **Value Calculation:** (`value_calculator.py`)
    * Determines player values (e.g., Value Over Replacement Player - VORP, or similar) based on consensus projections and active league settings.
* **Draft Engine/State Management:** (`draft_engine.py`)
    * Manages the state of a draft: available players, taken players (and by which team/pick), current pick.
* **Dynamic Value Updater:** (`dynamic_updater.py`)
    * Recalculates player values after each pick based on remaining players and team needs.
* **Draft Simulator:** (`draft_simulator.py`)
    * Runs simulated drafts using different strategies or ADP logic.
* **Output Generation:** (`output_generator.py`)
    * Saves processed dataframes (e.g., merged projections, value lists).
    * Generates reports or summaries.

### 1.3. MVP Definition
* **Part 1 (Core):** Successfully ingest, clean (basic), standardize (column names, team/pos abbreviations), normalize player names, and merge N expert projections into a single DataFrame. This DataFrame will have player names as the index (or a unique player ID) and columns for each expert's projection (e.g., `proj_expert1`, `proj_expert2`), along with standardized `Team` and `Position`.
* **Part 2 (Value Add):** Implement the projection scaling using historical data. Implement value calculation based on a defined league format.

### 1.4. Future Enhancements
* Automated download/acquisition of CSVs.
* Handling raw statistical projections (converting to fantasy points).
* Converting ranks to pseudo-projections.
* More sophisticated UI (web app, GUI).
* Advanced draft simulation parameters and analysis.
* Integration with live draft platforms (highly advanced).

## 2. Project Setup

### 2.1. Directory Structure
fantasy_football/
├── data/
│   ├── input/
│   │   └── {season_year}/         # e.g., 2025/
│   │       ├── projections/       # N expert projection CSVs
│   │       │   ├── expertA.csv
│   │       │   └── expertB.csv
│   │       └── historical/        # 1 historical actuals CSV
│   │           └── previous_year_actuals.csv # e.g., 2024_actuals.csv
│   ├── processed/
│   │   └── {season_year}/         # For intermediate and final dataframes
│   │       ├── cleaned_expertA.parquet
│   │       ├── merged_projections.parquet
│   │       └── player_values.parquet
│   └── output/
│       └── {season_year}/         # For reports, simulation results
│           └── draft_summary.txt
├── src/                           # Python source files
│   ├── init.py
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── player_normalizer.py
│   ├── projection_scaler.py
│   ├── data_merger.py
│   ├── league_manager.py
│   ├── value_calculator.py
│   ├── draft_engine.py
│   ├── dynamic_updater.py
│   ├── draft_simulator.py
│   ├── output_generator.py
│   ├── main.py
│   ├── config.py
│   └── utils.py
├── tests/                         # Unit tests
├── venv/                          # Virtual environment
├── .gitignore
├── requirements.txt
└── README.md                      # Overall project README
└── instructions.md                # This file (our detailed plan)

### 2.2. Key Configuration (`config.py` or `settings.yaml`)
* `CURRENT_SEASON_YEAR`: e.g., 2025 (used to determine data paths)
* `HISTORICAL_DATA_YEAR`: e.g., 2024 (derived from `CURRENT_SEASON_YEAR - 1`)
* Input data paths (constructable from `CURRENT_SEASON_YEAR`).
* Output data paths (constructable from `CURRENT_SEASON_YEAR`).
* List of "desired optional metadata columns" to attempt to carry through (e.g., `['Bye Week', 'Age']`).
* Default league settings (can be overridden).

---

## 3. Stage 1: Data Procurement & Loading (`data_loader.py`)

### 3.1. Requirements
* Ability to load multiple CSV files from a specified directory (`data/input/{season_year}/projections/`). The number of files (N) can vary.
* Ability to load a single CSV file from a specified directory (`data/input/{season_year}/historical/`).
* Store the source of each projection (e.g., filename or derived expert name) alongside the data.

### 3.2. Inputs
* `season_year` (from config or command line).
* Path to N projection CSV files: `data/input/{season_year}/projections/*.csv`
    * File naming is variable (e.g., `expertA_projections.csv`, `FantasySharks_Projections.csv`).
    * Each file represents projections from one distinct source.
    * Files contain columns for at least player name, team, position, and PPR projection points. Column names will vary.
    * Expected player count per file: 200-500.
* Path to 1 historical data CSV: `data/input/{season_year}/historical/previous_year_actuals.csv`
    * Contains actual stats and PPR points for players from the *previous* season.
    * Expected columns: `Player Name`, `Team`, `Pos`, `Games Played`, various stat categories, `Fantasy Points PPR`.

### 3.3. Outputs
* A list of Pandas DataFrames, where each DataFrame corresponds to one projection source. Each DataFrame should be augmented with a column indicating its source (e.g., 'expert_A', 'expert_B').
* One Pandas DataFrame for the historical data.

### 3.4. Process
1.  Read `CURRENT_SEASON_YEAR` from `config.py`.
2.  Construct input paths for projection and historical data.
3.  Identify all CSV files in the `projections` directory.
4.  For each projection CSV:
    * Load into a Pandas DataFrame.
    * Attempt to derive an "expert name" or "source identifier" from the filename (e.g., `expertA.csv` -> `expertA`). This identifier will be used as a prefix for projection columns later (e.g., `expertA_proj_ppr`). Store this identifier in a new column within the DataFrame (e.g., `source_expert`).
5.  Load the historical CSV into a Pandas DataFrame.
6.  Return the list of projection DataFrames and the single historical DataFrame.

### 3.5. Considerations & Edge Cases
* What if a projection CSV is empty or corrupt? (Log error, skip file, or halt?) -> For now, let's say log error and skip.
* What if the `projections` or `historical` directory is missing or empty? (Halt with clear error).
* Character encoding of CSVs: Assume UTF-8, but be aware other encodings might appear. (Pandas `read_csv` has an `encoding` parameter; might need to allow configuration or auto-detection if issues arise).
* Large files (unlikely for MVP, but good to note): Consider chunking if memory becomes an issue.

---

## 4. Stage 2: Data Cleaning & Standardization (`data_cleaner.py`)

### 4.1. Goal
To take the raw DataFrames from `data_loader.py` (for both projections and historical data) and:
1.  Standardize column names to a consistent internal representation.
2.  Clean and standardize team abbreviations.
3.  Clean and standardize position abbreviations.
4.  Perform basic data type cleaning (e.g., ensure projections are numeric, names are strings).
5.  Handle missing or problematic data gracefully (e.g., drop rows/files where essential data is missing, log issues).

### 4.2. Inputs
* A list of raw Pandas DataFrames (one per projection source), each augmented with a `source_expert` column.
* One raw Pandas DataFrame for historical data.
* Configuration files for:
    * Column name mappings (`column_mappings.yaml`).
    * Team abbreviation mappings (`team_mappings.yaml`).
    * Position abbreviation mappings (`position_mappings.yaml`).

### 4.3. Outputs
* A list of cleaned Pandas DataFrames (projection sources).
* One cleaned Pandas DataFrame (historical data).
* Each DataFrame will have:
    * Standardized column names (see 4.4.1).
    * New columns for standardized data (e.g., `cleaned_player_name`, `team_abbr_standardized`, `position_abbr_standardized`).
    * Original raw values for key fields preserved for traceability (e.g., `raw_player_name`, `raw_team_abbr`, `raw_pos_abbr`).
    * Cleaned data types (projections as numeric, names as lowercase strings).

### 4.4. Process Details

#### 4.4.1. Internal Standard Column Names:
* `source_expert`: Identifier for the projection source (from `data_loader.py`).
* `raw_player_name`: Original player name from the source file.
* `cleaned_player_name`: Player name after basic cleaning (lowercase, whitespace stripped). Input to `player_normalizer.py`.
* `raw_team_abbr`: Original team abbreviation/name from the source.
* `team_abbr_standardized`: Standardized team abbreviation (e.g., 'ARI', 'ATL') or `NaN` if unmappable.
* `raw_pos_abbr`: Original position abbreviation/name from the source.
* `position_abbr_standardized`: Standardized primary position (e.g., 'QB', 'RB') or `NaN` if unmappable.
* `projection_points_ppr`: Numeric projection points. Rows with `NaN` here will be dropped.
* `rank`: Numeric rank (optional, carried if present, converted to `NaN` if non-numeric).
* `metadata_{original_column_name}`: For any other desired optional metadata columns specified in `config.py` (e.g., `metadata_bye_week`, `metadata_age`). Original column name is preserved in the new name for clarity.

#### 4.4.2. Configuration File Examples:

* `column_mappings.yaml`:
    ```yaml
    # Maps lists of possible source column names to our internal concept
    # The cleaner will find the first existing column from the list and rename/use it.
    # These internal concepts will then be mapped to the final standardized column names.
    concepts:
      player_name: ['PLAYER NAME', 'Name', 'Full Name', 'Player']
      team: ['TEAM', 'Team', 'NFL Team']
      position: ['POS', 'Position', 'Depth Chart Position', 'Eligible Positions']
      projection_points: ['PROJ. PTS', 'PPR_Projection', 'FPts', 'PROJ_PPR', 'FantasyPointsPPR']
      rank: ['Rank', 'Overall Rank']
      bye_week: ['Bye', 'BYE Week', 'Bye Week']
      age: ['Age']
    # ... add more concepts for other desired optional metadata
    ```

* `team_mappings.yaml` (Key: Source Abbr, Value: Standardized Abbr):
    ```yaml
    # Standard is official NFL abbreviations
    JAX: JAX
    JAC: JAX
    STL: LAR # St. Louis Rams -> Los Angeles Rams
    NE: NE   # Standard
    NWE: NE  # New England
    NO: NO   # Standard
    NOR: NO  # New Orleans
    # ... comprehensive list
    ```

* `position_mappings.yaml` (Key: Source Abbr, Value: Standardized Abbr):
    ```yaml
    # Standard: QB, RB, WR, TE, K, DST
    QB: QB
    Q: QB
    RB: RB
    HB: RB
    FB: RB # Fullbacks mapped to RB
    WR: WR
    TE: TE
    PK: K
    K: K
    DST: DST
    DEF: DST
    D/ST: DST
    # ...
    ```

#### 4.4.3. Cleaning Steps (applied to each DataFrame):
1.  **Load Mappings:** Load all mapping configurations.
2.  **Column Name Standardization:**
    * For each internal concept (e.g., `player_name` concept from `column_mappings.yaml`):
        * Iterate through the list of possible source column names for that concept.
        * Find the first one that exists in the current DataFrame's columns.
        * Store this original column as `raw_{concept}` (e.g., copy `df['PLAYER NAME']` to `df['raw_player_name']`).
        * If the concept is `player_name`, create `cleaned_player_name` (see step 3).
        * If the concept is `team`, create `team_abbr_standardized` (see step 4).
        * If the concept is `position`, create `position_abbr_standardized` (see step 5).
        * If the concept is `projection_points`, rename the found column to `projection_points_ppr`.
        * For other concepts (rank, optional metadata), rename to `standardized_{concept}` or `metadata_{original_name}`.
    * If a column for `player_name` concept cannot be found: Log error for the file, and **skip processing this entire file/DataFrame**.
    * If a column for `projection_points` concept cannot be found: Log error for the file, and **skip processing this entire file/DataFrame**.
    * For other missing essential columns (team, position), log a warning; processing continues but relevant standardized columns will be `NaN`.
3.  **Player Name Cleaning (`cleaned_player_name`):**
    * Take the `raw_player_name` column.
    * Strip leading/trailing whitespace.
    * Convert to lowercase.
    * Store as `cleaned_player_name`.
4.  **Team Abbreviation Standardization (`team_abbr_standardized`):**
    * Take the `raw_team_abbr` column (derived from the mapped team concept column).
    * For each value:
        * Strip whitespace, convert to uppercase (to match keys in `team_mappings.yaml`).
        * Map using `team_mappings.yaml`.
        * If a team is not found in the mapping, log a warning (including the unmapped team name and source file) and set the value to `NaN`.
        * Store as `team_abbr_standardized`.
5.  **Position Abbreviation Standardization (`position_abbr_standardized`):**
    * Take the `raw_pos_abbr` column.
    * For each value:
        * Strip whitespace, convert to uppercase.
        * If multiple positions are listed (e.g., "RB,WR", "WR/TE"):
            * Split by common delimiters (',' '/', ';').
            * Take the *first* position in the list.
            * Log a warning if multiple positions were found, indicating which was chosen.
        * Map the chosen (or single) position using `position_mappings.yaml`.
        * If a position is not found in the mapping, log a warning (including the unmapped position and source file) and set the value to `NaN`.
        * Store as `position_abbr_standardized`.
6.  **Data Type Conversion and Cleaning:**
    * **`projection_points_ppr`**:
        * Remove any non-numeric characters (e.g., commas, asterisks) except decimal point and minus sign.
        * Convert to numeric (float). If conversion fails for a value, set to `NaN`.
    * **`rank`** (if present): Convert to numeric (integer or float if NaNs are present). If conversion fails, set to `NaN`.
    * Clean other optional metadata columns as appropriate (e.g., ensure `bye_week` is integer or `NaN`).
7.  **Row Filtering:**
    * Drop rows where `cleaned_player_name` is `NaN` or empty. Log these drops.
    * Drop rows where `projection_points_ppr` is `NaN`. Log these drops.
8.  **Select and Order Columns:** Ensure final DataFrames have a consistent set and order of columns (e.g., `source_expert`, `raw_player_name`, `cleaned_player_name`, `raw_team_abbr`, `team_abbr_standardized`, etc.).

#### 4.4.4. Logging Requirements:
* Files being processed.
* Successful column mappings (DEBUG level).
* Warnings:
    * Unmappable team abbreviations (value, source).
    * Unmappable position abbreviations (value, source).
    * Multiple positions found, primary one chosen (value, chosen, source).
    * Rows dropped due to missing critical data (name, projection points), with reason.
* Errors:
    * Files skipped due to missing essential columns (player name, projection points).
    * Corrupt files or files that cannot be parsed.

### 4.5. Considerations & Edge Cases
* **Iterative Refinement of Mappings:** The mapping YAML files will need to be updated as new data sources or new abbreviations are encountered. Good logging is key to identifying these.
* **Conflicting Mappings:** Ensure mapping keys are unique and well-defined.
* **Performance:** For the expected data sizes, direct string operations and map lookups in Pandas should be efficient enough.
* **Historical Data:** The same cleaning logic (column names, team/pos standardization, data types) must be applied to the historical data DataFrame. Team names in historical data might represent past franchises (e.g. "OAK" for Raiders before move to "LV"), so `team_mappings.yaml` needs to be comprehensive.

---

## 5. Stage 3: Player Name Normalization (`player_normalizer.py`)

### 5.1. Goal
To accurately map different string representations of player names from various sources (including historical data) to a single, consistent, **system-generated `canonical_player_id`**. This ensures that each unique real-world player is represented by one ID, allowing for correct aggregation and analysis. Any of the original source names can be used for display purposes later if needed, linked via this ID.

### 5.2. Unique Player Identifier Strategy

1.  **`canonical_player_id`**:
    * **Format**: A system-generated unique identifier (e.g., `player_0001`, `player_0002`, or a UUID). This ID does *not* need to be human-readable itself.
    * **Generation**: Assigned sequentially or via UUID when a new unique player entity is confirmed.

2.  **`master_player_roster.csv` (or `.parquet`)**:
    * A critical file that stores the established `canonical_player_id` and associated reference information. This file is loaded at the start and updated/saved at the end of the normalization process.
    * **Columns**:
        * `canonical_player_id` (Primary Key)
        * `display_name_example` (One of the `cleaned_player_name` values that mapped to this ID, e.g., the first one encountered or the longest. Used for reference/debugging.)
        * `known_team_abbr_standardized` (The most recently confirmed standardized team abbreviation for this player.)
        * `known_position_abbr_standardized` (The most recently confirmed standardized position for this player.)
        * `name_variants_seen` (Optional: a list or pipe-separated string of unique `cleaned_player_name` values mapped to this ID)
        * `suffix` (e.g., "jr", "iii", "sr" - normalized and stored if present, for display name construction)
        * `manually_confirmed` (Boolean: True if this ID was established or confirmed via `manual_overrides.csv`)
        * `date_added`
        * `last_updated`

3.  **`short_key` (Internal Matching Aid):**
    * To facilitate grouping and initial matching, a `short_key` will be generated for each player record from source data.
    * **Format**: `team_abbr_standardized + position_abbr_standardized + first_N_chars(normalize_name(cleaned_player_name))`
        * `normalize_name`: a function that removes all whitespace and special characters (e.g., '.', '-', "'") and converts to lowercase.
        * `first_N_chars`: e.g., first 3 or 4 characters of the normalized name. (Configurable: `N_SHORT_KEY_NAME_CHARS`)
    * Example: Kenneth Walker (RB, SEA) -> `cleaned_player_name`: "kenneth walker" -> normalized: "kennethwalker" -> `short_key` (N=3): "sear_bken" (assuming team 'SEA', pos 'RB').
        * Note: Standardized team/pos should be padded or fixed length if possible to ensure consistent key formatting (e.g., 3 chars for team, 2 for pos). Example: "SEA" + "RB" + "KEN".

### 5.3. Inputs
* A list of cleaned Pandas DataFrames (one per projection source) from `data_cleaner.py`.
    * Required columns: `cleaned_player_name`, `team_abbr_standardized`, `position_abbr_standardized`, `source_expert`, `raw_player_name`.
* One cleaned Pandas DataFrame for historical data.
* Configuration files:
    * `player_normalization_config.yaml`: Thresholds for fuzzy matching, N for `short_key` name chars.
    * `manual_overrides.csv`: User-maintained. Columns:
        * `input_raw_player_name` (for easy reference by user)
        * `input_cleaned_player_name` (for matching logic)
        * `input_team_abbr_standardized`
        * `input_position_abbr_standardized`
        * `input_source_expert` (Optional: to scope the override to a specific source)
        * `target_canonical_player_id` (The ID to assign. If this ID is new, it implies creating a new entry in master roster).
        * `target_display_name_example` (User-defined display name for this new ID, if creating new).
        * `comment` (User's notes)
* The `master_player_roster.csv` (if it exists from previous runs).

### 5.4. Outputs
* The input list of DataFrames (projections) and the historical DataFrame, each with a new column: `canonical_player_id`.
* An updated `master_player_roster.csv`.
* A `needs_review.csv` file. Columns:
    * `temp_group_id` (Identifier for a group of source rows that might be the same player but need review)
    * `source_expert`
    * `raw_player_name`
    * `cleaned_player_name`
    * `team_abbr_standardized`
    * `position_abbr_standardized`
    * `short_key_generated`
    * `suggested_canonical_player_id` (If any plausible candidate from master roster)
    * `match_score_to_suggestion` (If applicable)
    * `reason_for_review` (e.g., "New short_key group", "Ambiguous fuzzy match to master")

### 5.5. Core Matching Logic (Iterative, Prioritizing Confidence)

1.  **Initialization:**
    * Load `manual_overrides.csv`, `player_normalization_config.yaml`.
    * Load existing `master_player_roster.csv` into an in-memory DataFrame or dictionary for quick lookup.
    * For each player row in all input DataFrames, generate the `short_key`. Also, normalize suffixes (Jr., Sr., II, III) from `cleaned_player_name`, storing the base name and suffix separately.

2.  **Pass 1: Manual Overrides (Highest Precedence):**
    * Iterate through `manual_overrides.csv`. For each rule:
        * Find matching player(s) in the input DataFrames based on the `input_*` fields.
        * Assign the specified `target_canonical_player_id`.
        * If `target_canonical_player_id` is new to the `master_player_roster`, add it with the `target_display_name_example` and mark as `manually_confirmed=True`.
        * Remove successfully mapped players from further automated matching.

3.  **Pass 2: Exact Matches to Master Roster (using `short_key` and Name):**
    * For players not yet matched:
        * Attempt to match to the `master_player_roster` where:
            * The player's generated `short_key` matches a `short_key` derivable from the master roster's `display_name_example`, `known_team_abbr_standardized`, `known_position_abbr_standardized`.
            * AND the fuzzy name similarity (e.g., `thefuzz.ratio`) between the player's `cleaned_player_name` (base name) and the master roster's `display_name_example` (base name) is very high (e.g., >98, configurable).
        * If a unique, high-confidence match is found, assign the existing `canonical_player_id`. Update `master_player_roster` with latest team/pos if different and from a trusted current source. Remove from further matching.

4.  **Pass 3: Grouping by `short_key` & Intra-Group High-Confidence Fuzzy Matching (New Players):**
    * For players still not matched:
        * Group them by their generated `short_key`.
        * Within each group (players with the same `short_key`):
            * These players are highly likely to be the same person. Perform pairwise fuzzy name matching (e.g., `thefuzz.ratio()`) on their `cleaned_player_name` (base names).
            * If all names within the group are highly similar (e.g., all pairs >90, configurable threshold):
                * This group forms a new unique player.
                * Create a new `canonical_player_id`.
                * Select a `display_name_example` for the master roster (e.g., longest name in the group).
                * Add to `master_player_roster`.
                * Assign this new `canonical_player_id` to all players in this group. Remove from further matching.
            * If names within the `short_key` group are not sufficiently similar (e.g., "Ken Walker" and "Kevin Williams" somehow got same `short_key` due to collision):
                * These players/sub-groups within the `short_key` group are treated as distinct and move to `needs_review.csv`, flagged with their `short_key`.

5.  **Pass 4: Fuzzy Matching Remaining Players to Master Roster (Cautiously):**
    * For players *still* unmatched (likely have unique `short_key`s not forming clear groups, or failed intra-group similarity):
        * Attempt fuzzy match against the entire `master_player_roster` using `cleaned_player_name` (base), `team_abbr_standardized`, and `position_abbr_standardized`.
        * A strong match requires:
            * High name similarity (e.g., >90, configurable).
            * **AND** exact match on `team_abbr_standardized` AND `position_abbr_standardized`.
        * If such a unique match is found, assign the existing `canonical_player_id`. Update master roster. Remove from further matching.
        * If multiple master roster entries meet this, or if only name matches well but team/pos differ slightly, send to `needs_review.csv` with suggestions.

6.  **Pass 5: Handling Remaining Unmatched Players:**
    * Any players still unmatched at this point are likely new entities not seen before, or have very noisy data.
    * Group these remaining players by `cleaned_player_name`, `team_abbr_standardized`, `position_abbr_standardized`. If a group contains multiple source entries, they likely refer to a single new player.
    * For each such distinct new group/player:
        * Create a *new* `canonical_player_id`.
        * Select a `display_name_example`.
        * Add to `master_player_roster`.
        * Assign this new ID.
    * *Alternatively, for MVP, all players reaching this stage could be sent directly to `needs_review.csv` for manual `canonical_player_id` creation via `manual_overrides.csv`.* This is safer initially. 
    **Let's adopt this safer alternative for MVP: all players not matched by Pass 4 go to `needs_review.csv`.**

### 5.6. Suffix Normalization
* Handle suffixes like Jr., Sr., II, III, IV by:
    * Identifying and stripping them from `cleaned_player_name` to get a `base_name` for matching.
    * Storing the normalized suffix (e.g., "iii", "jr") in the `master_player_roster`.
    * The `display_name_example` in the master roster should ideally include the suffix if present.

### 5.7. Iteration & Review
* After each run, the user reviews `needs_review.csv`.
* Adds new rules to `manual_overrides.csv` to:
    * Correctly assign `canonical_player_id` to players in `needs_review.csv`.
    * Proactively define IDs for known tricky players.
* Re-running the normalizer will then use these new overrides.
* The `master_player_roster.csv` grows and becomes more accurate over time.

### 5.8. Configuration (`player_normalization_config.yaml`)
* `N_SHORT_KEY_NAME_CHARS`: Number of characters from the name to use in `short_key` (e.g., 3 or 4).
* Fuzzy match thresholds (0-100 scale):
    * `pass2_exact_to_master_name_similarity_threshold` (e.g., 98)
    * `pass3_intragroup_name_similarity_threshold` (e.g., 90)
    * `pass4_fuzzy_to_master_name_similarity_threshold` (e.g., 90)
* Paths to `manual_overrides.csv`, `master_player_roster.csv`.
* List of recognized player name suffixes.

### 5.9. Tools/Libraries
* **`thefuzz`** (and `python-Levenshtein`): For fuzzy string matching ratios.
* **Pandas:** For all data manipulation.
* **UUID library:** For generating `canonical_player_id` if UUIDs are chosen.

### 5.10. Key Considerations for MVP
* Focus on accurate implementation of `manual_overrides.csv` first.
* The `short_key` generation and matching (Pass 3) is crucial.
* High confidence thresholds for automated matches. When in doubt, send to `needs_review.csv`.
* Initial `master_player_roster.csv` might be empty or seeded from historical data if that data is considered clean.
* Sophisticated nickname handling can be deferred. Assume names are relatively standard.

---

## 6. Stage 4: Projection Scaling & Benchmarking (`projection_scaler.py`)

### 6.1. Goal
To adjust the raw PPR point projections from each expert source to a common scale. This is achieved by comparing each expert's projections for a defined set of top players at each key position against either:
    a) The actual historical performance of a similar set of players from the previous season.
    b) The average projection for that set of players across all current-year experts.
The aim is to mitigate biases from individual sources that might be consistently higher or lower in their overall projections, focusing on relative player values.

### 6.2. Inputs
* A list of Pandas DataFrames (one per projection source) from `player_normalizer.py`.
    * Required columns: `canonical_player_id`, `projection_points_ppr`, `team_abbr_standardized`, `position_abbr_standardized`, `source_expert`.
* One Pandas DataFrame for historical actuals (from `player_normalizer.py`).
    * Required columns: `canonical_player_id`, `fantasy_points_ppr` (previous season actuals), `position_abbr_standardized`.
* Configuration file: `projection_scaling_config.yaml`.

### 6.3. Outputs
* The same list of projection DataFrames, but each DataFrame will have a new column: `scaled_projection_points_ppr`.
    * The original `projection_points_ppr` column will be preserved.
    * A floor of 0 will be applied to `scaled_projection_points_ppr`.

### 6.4. Process Steps

1.  **Load Configuration (`projection_scaling_config.yaml`):**
    * `positions_to_benchmark`: List of positions to use for benchmarking (e.g., `['QB', 'RB', 'WR', 'TE']`).
    * `top_n_players_per_position`: Dictionary defining how many top players for each position form the benchmark set (e.g., `{'QB': 20, 'RB': 40, 'WR': 50, 'TE': 15}`).
    * `scaling_method`: Defines the target for scaling. Options:
        * `'historical'`: Scale expert projections to align with the previous year's actual totals for the benchmark player sets.
        * `'current_average'`: Scale expert projections to align with the average of all *current year* experts' projected totals for the benchmark player sets.
    * `min_benchmark_players_projected_threshold_pct`: (e.g., 0.75) - Minimum percentage of players from a historical benchmark set that an expert must project to calculate a reliable scaling factor for that expert/position. If below, use a default factor (e.g., 1.0) or an alternative strategy.

2.  **Identify Historical Benchmark Player Sets & Totals (for each position):**
    * For each `pos` in `positions_to_benchmark`:
        * Filter the historical DataFrame for players at that `pos`.
        * Sort these players by their previous season `fantasy_points_ppr` in descending order.
        * Select the top `N` players (where `N` is from `top_n_players_per_position[pos]`). These players (their `canonical_player_id`s) form the `historical_benchmark_player_set_{pos}`.
        * Calculate the sum of `fantasy_points_ppr` for this set: `historical_benchmark_total_{pos}`.
        * Store these sets and totals.

3.  **Calculate Expert Projected Totals for Benchmark Sets (for each expert, for each position):**
    * Initialize a data structure to store `expert_projected_total_for_benchmark_set_{expert}_{pos}`.
    * For each `expert_df` in the list of projection DataFrames:
        * For each `pos` in `positions_to_benchmark`:
            * Get the `historical_benchmark_player_set_{pos}`.
            * Filter `expert_df` to include only players who are in this `historical_benchmark_player_set_{pos}` (matching on `canonical_player_id`).
            * Count how many players from the benchmark set were found in the expert's projections (`count_projected_benchmark_players`).
            * If `count_projected_benchmark_players / N < min_benchmark_players_projected_threshold_pct[pos]`:
                * Log a warning for this expert/position. The expert doesn't project enough of the benchmark players for a reliable direct scaling factor.
                * This expert's projections for this `pos` might use a default scaling factor of 1.0 or an aggregated factor (see step 4). For now, let's assume it will default to 1.0 for this specific position for this expert if a reliable factor cannot be computed.
            * Else (enough players projected):
                * Sum the `projection_points_ppr` for these players from `expert_df`. This is `expert_projected_total_for_benchmark_set_{expert}_{pos}`.

4.  **Determine Position-Specific Scaling Factors for Each Expert:**
    * **If `scaling_method == 'historical'`:**
        * For each expert and each `pos`:
            * If `expert_projected_total_for_benchmark_set_{expert}_{pos}` was reliably calculated:
                * `scaling_factor_{expert}_{pos} = historical_benchmark_total_{pos} / expert_projected_total_for_benchmark_set_{expert}_{pos}`
            * Else (not reliably calculated):
                * `scaling_factor_{expert}_{pos} = 1.0` (default, no scaling for this segment)
    * **If `scaling_method == 'current_average'`:**
        * First, for each `pos`, calculate the `average_all_experts_projected_total_for_benchmark_set_{pos}`:
            * Collect all valid (reliably calculated) `expert_projected_total_for_benchmark_set_{expert}_{pos}` values for that `pos`.
            * Calculate their average.
        * Then, for each expert and each `pos`:
            * If `expert_projected_total_for_benchmark_set_{expert}_{pos}` was reliably calculated:
                * `scaling_factor_{expert}_{pos} = average_all_experts_projected_total_for_benchmark_set_{pos} / expert_projected_total_for_benchmark_set_{expert}_{pos}`
            * Else (not reliably calculated):
                * `scaling_factor_{expert}_{pos} = 1.0` (default)
    * Handle potential division by zero if an `expert_projected_total_for_benchmark_set` is zero (factor becomes 1.0 or a large cap if numerator is non-zero).

5.  **Apply Scaling Factors to Projections:**
    * For each `expert_df`:
        * Create a new `scaled_projection_points_ppr` column, initially a copy of `projection_points_ppr`.
        * For each `pos` in `positions_to_benchmark`:
            * Get the `scaling_factor_{expert}_{pos}`.
            * For all players in `expert_df` whose `position_abbr_standardized == pos`, multiply their `scaled_projection_points_ppr` by this factor.
        * For players in `expert_df` whose `position_abbr_standardized` is *not* in `positions_to_benchmark` (e.g., 'K', 'DST' if not benchmarked):
            * Their `scaled_projection_points_ppr` remains unadjusted by this positional scaling (i.e., factor of 1.0). Alternatively, an average scaling factor for the expert could be applied here if desired (config option for future). For MVP, let's keep them unscaled if their position is not benchmarked.
        * Apply a floor of 0 to all values in `scaled_projection_points_ppr`.

### 6.5. Logging & Traceability
* Log the `historical_benchmark_player_set_{pos}` (e.g., list of IDs or count) and `historical_benchmark_total_{pos}` for each benchmarked position.
* For each expert and position, log the `expert_projected_total_for_benchmark_set`, number of benchmark players found, and the calculated `scaling_factor`.
* Log warnings for experts/positions where reliable scaling factors couldn't be determined and default (1.0) was used.

### 6.6. Configuration Example (`projection_scaling_config.yaml`)
```yaml
positions_to_benchmark: ['QB', 'RB', 'WR', 'TE']

top_n_players_per_position:
  QB: 20 # Top 20 QBs from last year form the QB benchmark set
  RB: 20
  WR: 64
  TE: 12

# Scaling method: 'historical' or 'current_average'
scaling_method: 'historical'

# Minimum percentage of players from a historical benchmark set
# that an expert must project for that position to calculate a scaling factor.
# If below this, a factor of 1.0 (no scaling) is used for that expert/position.
min_benchmark_players_projected_threshold_pct: 0.75 # 75%

# Policy for positions not explicitly benchmarked (e.g., K, DST)
# Options: 'no_scale' (factor of 1.0) or 'apply_average_expert_factor' (future enhancement)
unbenchmarked_position_policy: 'no_scale'
```

### 6.7. Edge Cases & Considerations
* Player Position Changes: A player might be in the historical benchmark set as an 'RB' but projected as a 'WR' by a current expert. The scaling factor applied to this player by the expert should be based on their current projected position in that expert's list. The benchmark sets are purely from historical data to define a stable group of players and their aggregate scores.
* Very Few Players Projected by an Expert: If an expert provides a very shallow list, they might not cover enough players in the benchmark sets. The min_benchmark_players_projected_threshold_pct helps handle this by defaulting to no scaling for that expert/position, preventing extreme scaling factors based on small samples.
* Impact of Player Pool Depth: The choice of top_n_players_per_position can influence results. These should be reasonable numbers representing established, consistently fantasy-relevant players.
* Zero Projections: If an expert projects 0 points for a benchmark set (that historical or others project points for), the scaling factor could become infinite or very large. Apply a cap to scaling factors (e.g., max factor of 2.0, min of 0.5 - configurable) or handle division by zero by defaulting factor to 1.0.
* Alternative to Position-Specific Factors: While position-specific factors are implemented here, a simpler future alternative could be to calculate one aggregate scaling factor per expert (e.g., by averaging position-specific factors, perhaps weighted by total points in each benchmark group). For now, position-specific provides more targeted adjustments.

---

## 7. Stage 5: Data Merging (`data_merger.py`)

### 7.1. Goal
To consolidate all processed and scaled player projections from the various expert sources into a single, unified DataFrame. This DataFrame will serve as the primary output for this phase of the project, providing a clear overview of consensus (and individual expert) projections for each uniquely identified player.

### 7.2. Inputs
* A list of Pandas DataFrames (one per projection source), output by `projection_scaler.py`.
    * Each DataFrame contains: `canonical_player_id`, `scaled_projection_points_ppr`, `projection_points_ppr` (original raw projection), `source_expert` (identifier for the expert), and other player metadata like `team_abbr_standardized`, `position_abbr_standardized` as determined by that expert for that player.
* The `master_player_roster.csv` file (output by `player_normalizer.py`).
    * Contains: `canonical_player_id`, `display_name_example`, `known_team_abbr_standardized`, `known_position_abbr_standardized`, `suffix`.
* Configuration from `config.py` or a dedicated `data_merging_config.yaml`:
    * `output_filename_merged_projections` (e.g., `"merged_player_projections.parquet"` or `"merged_player_projections.csv"`)
    * `include_original_projections_in_merged_output` (boolean, e.g., `True`)
    * Base output directory (e.g., `data/processed/{season_year}/`)

### 7.3. Outputs
* **A single Pandas DataFrame (`merged_df`) with the following structure:**
    * **Index:** `display_name` (taken from `master_player_roster.display_name_example` for the corresponding `canonical_player_id`).
    * **Columns:**
        * `player_id`: The `canonical_player_id`.
        * `team`: The `known_team_abbr_standardized` from the `master_player_roster`.
        * `position`: The `known_position_abbr_standardized` from the `master_player_roster`.
        * `proj_scaled_{EXPERT_ID_1}`: Scaled PPR projection from Expert 1.
        * `proj_scaled_{EXPERT_ID_2}`: Scaled PPR projection from Expert 2.
        * ... (for all N experts)
        * (Optional, if `include_original_projections_in_merged_output` is True)
            * `proj_orig_{EXPERT_ID_1}`: Original PPR projection from Expert 1.
            * `proj_orig_{EXPERT_ID_2}`: Original PPR projection from Expert 2.
            * ... (for all N experts)
* This `merged_df` is saved locally to the path specified in the configuration (e.g., `data/processed/{season_year}/merged_player_projections.parquet`).

### 7.4. Process Steps

1.  **Load Master Player Roster:**
    * Read `master_player_roster.csv` into a DataFrame (`master_roster_df`).
    * Select necessary columns: `canonical_player_id`, `display_name_example`, `known_team_abbr_standardized`, `known_position_abbr_standardized`.
    * Rename columns for clarity in the final output if needed (e.g., `display_name_example` to `display_name`, `known_team_abbr_standardized` to `team`, `known_position_abbr_standardized` to `position`).
    * Set `canonical_player_id` as the index of `master_roster_df`. This will be the base for our merged data.

2.  **Prepare and Merge Each Expert's Projections:**
    * Initialize `merged_projections_df = master_roster_df.copy()`.
    * For each `expert_df` in the list of processed projection DataFrames (from `projection_scaler.py`):
        * Get the `source_expert` identifier (e.g., "expertA", "FantasyPros"). Sanitize this identifier to ensure it's a valid and clean column name suffix (e.g., replace spaces or special characters with underscores, remove ".csv").
        * Create a temporary DataFrame containing:
            * `canonical_player_id`
            * `proj_scaled_{source_expert}` (renamed from `scaled_projection_points_ppr`)
            * If `include_original_projections_in_merged_output` is True:
                * `proj_orig_{source_expert}` (renamed from `projection_points_ppr`)
        * Set `canonical_player_id` as the index for this temporary expert projection DataFrame.
        * Perform a **left merge** of `merged_projections_df` with this temporary expert projection DataFrame, using `canonical_player_id` (the index) as the merge key.
            * `merged_projections_df = pd.merge(merged_projections_df, temp_expert_proj_df, left_index=True, right_index=True, how='left')`
            * A left merge ensures that all players from the `master_player_roster` are kept. If an expert doesn't have a projection for a player in the master roster, their column(s) will have `NaN` for that player.

3.  **Finalize DataFrame Structure:**
    * Reset the index of `merged_projections_df` if `canonical_player_id` was the index, so `canonical_player_id` becomes a regular column.
    * Set `display_name` as the index, as per the desired output structure ("index 'name'").
    * Rename `canonical_player_id` column to `player_id` for the final output.
    * Ensure desired column order: `player_id`, `team`, `position`, followed by all projection columns.

4.  **Save Output:**
    * Construct the full output path using the base output directory, current season year, and configured filename.
    * Save `merged_projections_df` to this path. Parquet format is recommended for efficiency and type preservation, but CSV is also an option.
        * Example: `merged_projections_df.to_parquet(output_path)`

### 7.5. Configuration Notes (`config.py` or `data_merging_config.yaml`)
```python
# Example in config.py
MERGED_PROJECTIONS_FILENAME = "merged_player_projections.parquet"
MERGED_PROJECTIONS_INCLUDE_ORIGINAL = True # or False
# output_dir_processed = f"data/processed/{CURRENT_SEASON_YEAR}/" (defined elsewhere)
```

### 7.6. Simplicity Notes & Edge Cases
* Core Simplicity: This process relies heavily on the canonical_player_id being accurate and present in all DataFrames. The merging logic itself is reduced to standard Pandas joins.
* NaN Values: Expect NaN values in expert projection columns where an expert did not provide a projection for a particular player listed in the master roster. This is normal and correctly handled by outer/left joins.
* Completeness of Master Roster: If a player was projected by an expert but somehow missed being added to the master_player_roster during normalization (unlikely if logic is correct), they would not appear in the final merged_df if master_player_roster is the definitive base. The current approach of left-merging onto the master roster ensures all players deemed "canonical" are present.

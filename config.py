
"""
Fantasy Football Projection Aggregator & Draft Tool - Configuration

This file contains all configuration settings for the Fantasy Football Projection Aggregator & Draft Tool.
"""
import os
from pathlib import Path
from datetime import datetime

# ===============================
# GENERAL SETTINGS
# ===============================

# Active season year (the year for which projections are being analyzed)
SEASON_YEAR = 2025
HISTORICAL_YEAR = SEASON_YEAR - 1

DRAFT_POSITION = 5
N_TEAMS = 10
ROSTER_N_QB = 1
ROSTER_N_RB = 2
ROSTER_N_WR = 2
ROSTER_N_TE = 1

ROSTER_N_FLEX = 2
ROSTER_FLEX_ELIGIBLE_POSITIONS = ["RB", "WR", "TE"]

ROSTER_N_K = 0
ROSTER_N_DST = 0
ROSTER_N_BENCH = 7

PROJECTION_COLUMN_PREFIX = "projection_points_ppr"
POSITION_COLUMN = "position_abbr_standardized"

# Base directory paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"

# Input data paths
INPUT_DIR = DATA_DIR / "input" / str(SEASON_YEAR)
PROJECTIONS_DIR = INPUT_DIR / "projections"
HISTORICAL_DIR = INPUT_DIR / "historical"

# Processed data paths
PROCESSED_DIR = DATA_DIR / "processed" / str(SEASON_YEAR)
OUTPUT_DIR = DATA_DIR / "output" / str(SEASON_YEAR)

# Create directories if they don't exist
for dir_path in [PROJECTIONS_DIR, HISTORICAL_DIR, PROCESSED_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Logging configuration
LOG_DIR = BASE_DIR / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = LOG_DIR / f"ff_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# ===============================
# LEAGUE SETTINGS
# ===============================

# Default league settings (can be overridden at runtime)
DEFAULT_LEAGUE_SETTINGS = {
    "name": "Default PPR League",
    "teams": N_TEAMS,
    "roster_spots": {
        "QB": ROSTER_N_QB,
        "RB": ROSTER_N_RB,
        "WR": ROSTER_N_WR,
        "TE": ROSTER_N_TE,
        "FLEX": ROSTER_N_FLEX,
        "K": ROSTER_N_K,
        "DST": ROSTER_N_DST,
        "BENCH": ROSTER_N_BENCH
    },
    "flex_eligible_positions": {
        "FLEX": ROSTER_FLEX_ELIGIBLE_POSITIONS
    },
    "scoring": {
        "passing_yards": 0.04,  # 1 point per 25 yards
        "passing_td": 4,
        "interception": -1,
        "rushing_yards": 0.1,  # 1 point per 10 yards
        "rushing_td": 6,
        "receiving_yards": 0.1,  # 1 point per 10 yards
        "reception": 1.0,  # PPR
        "receiving_td": 6,
        "fumble_lost": -2,
        "two_point_conversion": 2
    },
    "draft_type": "snake",  # Options: "snake", "auction", "linear"
    "draft_position": DRAFT_POSITION,  # User's draft position (1-based)
}

# ===============================
# DATA LOADING SETTINGS
# ===============================

# File extensions to consider when loading projection files
VALID_FILE_EXTENSIONS = [".csv"]

# Historical data filename pattern
HISTORICAL_FILE_PATTERN = "*_actuals.csv"

# ===============================
# DATA CLEANING SETTINGS
# ===============================

PLAYER_NAME_MAPPINGS = {
    'cam ward': 'cameron ward',
    'cam skattebo': 'cameron skattebo',
    'josh palmer': 'joshua palmer',
}

# Column name standardization mappings
# Maps various column names from different sources to our standard names
COLUMN_NAME_MAPPINGS = {
    # Player name variations
    "player": "player_name",
    "player name": "player_name",
    "name": "player_name",
    "full name": "player_name",
    
    # Team variations
    "team": "team_abbr",
    "tm": "team_abbr",
    "nfl team": "team_abbr",
    
    # Position variations
    "position": "position_abbr",
    "pos": "position_abbr",
    "depth chart position": "position_abbr",
    
    # Projection variations
    "proj. pts": PROJECTION_COLUMN_PREFIX,
    "ppr_projection": PROJECTION_COLUMN_PREFIX,
    "fpts": PROJECTION_COLUMN_PREFIX,
    "ppr": PROJECTION_COLUMN_PREFIX,
    "proj_ppr": PROJECTION_COLUMN_PREFIX,
    "points": PROJECTION_COLUMN_PREFIX,
    "fantasy points": PROJECTION_COLUMN_PREFIX,
    "projection": PROJECTION_COLUMN_PREFIX,
    "projected points": PROJECTION_COLUMN_PREFIX,
    
    # Rank variations
    "rank": "rank",
    "overall rank": "rank",
    "ovr": "rank",
    "overall_rank": "rank",

    # position rank variations
    "pos_rank": "position_rank",
    "pos_ovr": "position_rank",
    "pos_ovr_rank": "position_rank",
    "position rank": "position_rank",

    # ADP variations
    "adp": "adp",
    "adp_rank": "adp_rank",
    "adp_ovr": "adp_rank",
}

# Team abbreviation standardization
# Maps various team abbreviations to standard NFL team codes
TEAM_ABBR_MAPPINGS = {
    # AFC East
    "buf": "BUF", "buffalo": "BUF", "bills": "BUF",
    "mia": "MIA", "miami": "MIA", "dolphins": "MIA",
    "ne": "NE", "new england": "NE", "patriots": "NE", "nwe": "NE",
    "nyj": "NYJ", "new york jets": "NYJ", "jets": "NYJ",
    
    # AFC North
    "bal": "BAL", "blt": "BAL", "ravens": "BAL",
    "cin": "CIN", "cincinnati": "CIN", "bengals": "CIN",
    "cle": "CLE", "cleveland": "CLE", "browns": "CLE", "clv": "CLE",
    "pit": "PIT", "pittsburgh": "PIT", "steelers": "PIT",
    
    # AFC South
    "hou": "HOU", "houston": "HOU", "texans": "HOU", "hst": "HOU",
    "ind": "IND", "indianapolis": "IND", "colts": "IND",
    "jac": "JAX", "jacksonville": "JAX", "jaguars": "JAX", "jax": "JAX",
    "ten": "TEN", "tennessee": "TEN", "titans": "TEN",
    
    # AFC West
    "den": "DEN", "denver": "DEN", "broncos": "DEN",
    "kc": "KC", "kansas city": "KC", "chiefs": "KC", "kan": "KC",
    "lv": "LV", "las vegas": "LV", "raiders": "LV", "oak": "LV", "oakland": "LV", "lvr": "LV",
    "lac": "LAC", "la chargers": "LAC", "los angeles chargers": "LAC", "chargers": "LAC", "sd": "LAC", "san diego": "LAC",
    
    # NFC East
    "dal": "DAL", "dallas": "DAL", "cowboys": "DAL",
    "nyg": "NYG", "new york giants": "NYG", "giants": "NYG",
    "phi": "PHI", "philadelphia": "PHI", "eagles": "PHI",
    "was": "WAS", "washington": "WAS", "commanders": "WAS", "football team": "WAS", "redskins": "WAS",
    
    # NFC North
    "chi": "CHI", "chicago": "CHI", "bears": "CHI",
    "det": "DET", "detroit": "DET", "lions": "DET",
    "gb": "GB", "green bay": "GB", "packers": "GB", "gnb": "GB",
    "min": "MIN", "minnesota": "MIN", "vikings": "MIN",
    
    # NFC South
    "atl": "ATL", "atlanta": "ATL", "falcons": "ATL",
    "car": "CAR", "carolina": "CAR", "panthers": "CAR",
    "no": "NO", "new orleans": "NO", "saints": "NO", "nor": "NO",
    "tb": "TB", "tampa bay": "TB", "buccaneers": "TB", "bucs": "TB", "tam": "TB",
    
    # NFC West
    "ari": "ARI", "arizona": "ARI", "cardinals": "ARI", "az": "ARI", "arz": "ARI",
    "la": "LAR", "los angeles": "LAR", "rams": "LAR", "lar": "LAR",
    "sf": "SF", "san francisco": "SF", "49ers": "SF", "sfo": "SF",
    "sea": "SEA", "seattle": "SEA", "seahawks": "SEA",
    
    # Free agents
    "fa": "FA", "free agent": "FA", "none": "FA", "": "FA", "nan": "FA", "uns": "FA",

    # multiple teams (for historical data)
    "multiple": "MULT", "mult": "MULT", "multiple teams": "MULT", "2tm": "MULT", "3tm": "MULT",
}

# Position abbreviation standardization
POSITION_ABBR_MAPPINGS = {
    "quarterback": "QB", "qb": "QB",
    "running back": "RB", "rb": "RB", "halfback": "RB", "hb": "RB", "fb": "RB",
    "wide receiver": "WR", "wr": "WR",
    "tight end": "TE", "te": "TE",
    "kicker": "K", "k": "K", "pk": "K", "placekicker": "K",
    "defense": "DST", "dst": "DST", "DEF": "DST", "team defense": "DST", "def": "DST",
    "defensive line": "DL", "dl": "DL",
    "linebacker": "LB", "lb": "LB",
    "defensive back": "DB", "db": "DB",
    "individual defensive player": "IDP", "idp": "IDP",
}

# List of desired optional metadata columns to carry through
DESIRED_OPTIONAL_METADATA = [
    "bye_week",
    "age",
    "adp"
]

# ===============================
# PLAYER NAME NORMALIZATION SETTINGS
# ===============================

# Number of characters from name to use in short_key for initial matching
N_SHORT_KEY_NAME_CHARS = 4

# Fuzzy matching thresholds
FUZZY_MATCH_THRESHOLD_HIGH = 90  # Threshold for high-confidence matches (0-100)
FUZZY_MATCH_THRESHOLD_MEDIUM = 80  # Threshold for medium-confidence matches (0-100)

# ===============================
# PROJECTION SCALING SETTINGS
# ===============================

# Scaling method options: 'historical', 'consensus', 'none'
SCALING_METHOD = "historical"

# Number of top players per position to use as benchmark set
BENCHMARK_PLAYER_COUNTS = {
    "QB": 16,
    "RB": 24,
    "WR": 40,
    "TE": 15,
    # "K": 10,
    # "DST": 32
}

# Limits on scaling factors to prevent extreme adjustments
MIN_SCALING_FACTOR = 0.5
MAX_SCALING_FACTOR = 2.0

# ===============================
# VALUE CALCULATION SETTINGS
# ===============================

# Value calculation method: 'vorp', 'var', 'custom'
VALUE_CALCULATION_METHOD = "vorp"

# Baseline type for VORP: 'positional_scarcity', 'last_starter', 'custom'
VORP_BASELINE_TYPE = "positional_scarcity"

# ===============================
# DRAFT SIMULATION SETTINGS
# ===============================

# Number of simulations to run
NUM_SIMULATIONS = 100

# Draft strategy options for AI teams
AI_DRAFT_STRATEGIES = [
    "best_available",  # Pick highest value player regardless of position
    "need_based",      # Prioritize positions of need
    "value_based",     # Balance value and need
    "zero_rb",         # Avoid RBs in early rounds
    "robust_rb",       # Prioritize RBs in early rounds
    "balanced"         # Balanced approach across positions
]

# Default strategy distribution for AI teams
DEFAULT_AI_STRATEGY_DISTRIBUTION = {
    "best_available": 0.3,
    "need_based": 0.2,
    "value_based": 0.3,
    "zero_rb": 0.1,
    "robust_rb": 0.1,
    "balanced": 0.0  # Disabled by default
}

# ===============================
# OUTPUT SETTINGS
# ===============================

# File formats for saving processed data
PROCESSED_FILE_FORMAT = "pickle"  # Options: "csv", "parquet", "pickle"

# Report generation settings
GENERATE_SUMMARY_REPORT = True
GENERATE_DRAFT_CHEATSHEET = True
GENERATE_TEAM_ANALYSIS = True
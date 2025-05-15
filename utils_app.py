import pandas as pd
import config

# --- constants you can tweak ---------------------------------
MY_DRAFT_PICK = config.DRAFT_POSITION
N_TEAMS = config.N_TEAMS

ADD_KEEPERS = True
KEEPERS = {'breece hall': 21,
           'terry mclaurin': 50,
           'jayden daniels': 74,
           'ladd mcconkey': 77,
           'jaxon smith-njigba': 78,
           'chuba hubbard': 99,
           "de'von achane": 105,
           'puka nacua': 132,
           'kyren williams': 133,
           'rashid shaheed': 135}

VISIBLE_EDIT_COLS = ["team", "position", "drafted"]
INT_COLS          = {"drafted", "rank", "rank_pos", "rank_pos_team"}

POS_COLORS = {
    "QB": "#d0e7ff",   # light blue
    "RB": "#d7f9d7",   # light green
    "WR": "#eadcff",   # light purple
    "TE": "#ffe9d6",   # light orange
}

def round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    num_cols = df2.select_dtypes("number").columns.difference(INT_COLS)
    # df2[num_cols] = df2[num_cols].round(1)
    df2[num_cols] = df2[num_cols].astype(int) # todo: place holder, working on this, should be one decimal
    df2[list(INT_COLS)] = df2[list(INT_COLS)].astype(int)
    return df2

def position_tint(row):
    color = POS_COLORS.get(row["position"], "")
    return [f"background-color: {color}" for _ in row]

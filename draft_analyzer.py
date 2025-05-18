# draft_analyzer.py
import pandas as pd
from typing import List, Dict, Tuple
import config  # your settings module

# ---------- helpers ----------
def pick_owner(pick: int, n_teams: int) -> int:
    """Return drafting slot (1-indexed) that owns overall pick `pick`."""
    rd = (pick - 1) // n_teams          # 0-indexed round
    pos = (pick - 1) % n_teams + 1      # slot in that round
    return pos if rd % 2 == 0 else n_teams - pos + 1

# ---------- core ----------
class DraftAnalyzer:
    FLEX_POS = set(config.ROSTER_FLEX_ELIGIBLE_POSITIONS)

    def __init__(self, live_df: pd.DataFrame):
        self.df = live_df.copy()
        self.df["owner"] = self.df["drafted"].apply(
            lambda p: pick_owner(p, config.N_TEAMS) if p > 0 else None
        )
        self.my_df = self.df[self.df["owner"] == config.DRAFT_POSITION]
        self.lineup_slots = self._build_slot_list()

    # ----- public API -----
    def roster_table(self) -> pd.DataFrame:
        """Return table indexed by QB1, RB1 … BNn."""
        starters, bench = self._split_starters_bench()
        table = pd.concat([starters, bench]).reindex(self.lineup_slots)
        return table[["player", "position", "team", "rank_pos",
                      "median_projection", "static_value", "round_drafted"]]

    def pos_sums(self) -> Dict[str, float]:
        """Median-projection sums for my starting groups."""
        starters, _ = self._split_starters_bench()
        return starters.groupby("slot_group")["median_projection"].sum().to_dict()

    def strength_scores(self):
        my_sums = self.pos_sums()

        # ----- A. pct vs theoretical max -----
        top_pool = self.df[self.df["drafted"] > 0] \
            .sort_values("median_projection", ascending=False)
        max_sums = {
            g: top_pool[top_pool["position"].isin(self._group_positions(g))]  # ok if empty
            .head(self._group_n(g))["median_projection"].sum()
            for g in my_sums
        }
        pct_max = {g: (my_sums[g] / max_sums[g]) if max_sums[g] else 0.0
                   for g in my_sums}

        # ----- B. pct vs best team so far -----
        team_best = {g: 0.0 for g in my_sums}
        for tm in range(1, config.N_TEAMS + 1):
            t_df = self.df[self.df["owner"] == tm]
            t_sum = self._group_sums_for_team(t_df)  # may omit keys
            for g in my_sums:
                team_best[g] = max(team_best[g], t_sum.get(g, 0.0))

        pct_league = {g: (my_sums[g] / team_best[g]) if team_best[g] else 1.0
                      for g in my_sums}

        return pct_max, pct_league

    # ----- internals -----
    def _build_slot_list(self) -> List[str]:
        bn = [f"BN{i+1}" for i in range(config.ROSTER_N_BENCH)]
        return (
            ["QB1"] +
            [f"RB{i+1}" for i in range(config.ROSTER_N_RB)] +
            [f"WR{i+1}" for i in range(config.ROSTER_N_WR)] +
            ["TE1"] +
            [f"FLEX{i+1}" for i in range(config.ROSTER_N_FLEX)] +
            (["DST1"] if config.ROSTER_N_DST else []) +
            (["K1"] if config.ROSTER_N_K else []) +
            bn
        )

    def _split_starters_bench(self):
        df = self.my_df.sort_values("median_projection", ascending=False).copy()
        df["round_drafted"] = (df["drafted"] - 1) // config.N_TEAMS + 1
        df["slot_group"] = None

        slots_left = {
            "QB": config.ROSTER_N_QB, "RB": config.ROSTER_N_RB,
            "WR": config.ROSTER_N_WR, "TE": config.ROSTER_N_TE,
            "DST": config.ROSTER_N_DST, "K": config.ROSTER_N_K,
            "FLEX": config.ROSTER_N_FLEX
        }

        starters_rows, starter_idx = [], []
        for idx, row in df.iterrows():
            pos = row["position"]
            grp = None
            if slots_left.get(pos, 0):
                grp, slots_left[pos] = pos, slots_left[pos] - 1
            elif pos in self.FLEX_POS and slots_left["FLEX"]:
                grp, slots_left["FLEX"] = "FLEX", slots_left["FLEX"] - 1

            if grp:
                row["slot_group"] = grp
                starters_rows.append(row)
                starter_idx.append(idx)          # track originals

        starters_df = (
            pd.DataFrame(starters_rows)
              .assign(slot=lambda x: x.groupby("slot_group").cumcount() + 1)
              .assign(index=lambda x: x.apply(
                  lambda r: f"{r.slot_group}{r.slot}" if r.slot_group != "QB" else "QB1",
                  axis=1))
              .set_index("index")
        )

        # use original row indices to exclude starters
        bench_df = df.loc[~df.index.isin(starter_idx)].copy()
        bench_df["index"] = [f"BN{i+1}" for i in range(len(bench_df))]
        bench_df = bench_df.set_index("index")

        return starters_df, bench_df

    def _group_positions(self, g):
        return {"RB": ["RB"], "WR": ["WR"], "FLEX": list(self.FLEX_POS),
                "QB": ["QB"], "TE": ["TE"], "DST": ["DST"], "K": ["K"]}[g]

    def _group_n(self, g):
        return {
            "QB": config.ROSTER_N_QB, "RB": config.ROSTER_N_RB,
            "WR": config.ROSTER_N_WR, "TE": config.ROSTER_N_TE,
            "FLEX": config.ROSTER_N_FLEX,
            "DST": config.ROSTER_N_DST, "K": config.ROSTER_N_K
        }[g]

    def _group_sums_for_team(self, team_df):
        self_tmp = self.my_df              # keep attr intact
        self.my_df = team_df
        sums = self.pos_sums()
        self.my_df = self_tmp
        return sums

if __name__ == "__main__":

    ''' ==================================================================
        SETUP
    ================================================================== '''
    from gen_values import get_raw_df, value_players

    # Keeper settings
    ADD_KEEPERS = True
    KEEPERS = {
        'breece hall': 21,
        'terry mclaurin': 50,
        'jayden daniels': 74,
        'ladd mcconkey': 77,
        'jaxon smith-njigba': 78,
        'chuba hubbard': 99,
        "de'von achane": 105,
        'puka nacua': 132,
        'kyren williams': 133,
        'rashid shaheed': 135
    }

    raw_df = get_raw_df()

    # Mark keepers as drafted if enabled
    if ADD_KEEPERS:
        for player, pick_num in KEEPERS.items():
            raw_df.loc[raw_df['player'] == player, 'drafted'] = pick_num

    live_draft_board = value_players(df=raw_df)

    ''' ==================================================================
        EXAMPLE DRAFT ANALYZER
    ================================================================== '''
    lineup = DraftAnalyzer(live_draft_board)

    # 1️⃣ table
    roster_df = lineup.roster_table()

    # 2️⃣ my starter sums
    pos_totals = lineup.pos_sums()

    # 3️⃣ strength scores
    pct_max, pct_league = lineup.strength_scores()

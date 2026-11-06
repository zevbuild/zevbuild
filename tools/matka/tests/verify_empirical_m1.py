"""
Empirical Verification & Verification Data Table Generator for Milestone 1.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MATKA_ROOT = REPO_ROOT / "tools" / "matka"
MATKA_ENGINE = MATKA_ROOT / "kalyan_4_35_to_6_35"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MATKA_ENGINE))

from models import EnsemblePredictor, get_family_jodis

def main():
    csv_file = MATKA_ROOT / "kalyan_historical_data.csv"
    df = pd.read_csv(csv_file)
    valid_df = df[df["Is_Valid"] == True].copy().reset_index(drop=True)

    predictor = EnsemblePredictor()
    predictor.fit(valid_df)

    ref_draws = [
        {"date": "2026-10-26", "day": "Mon", "actual_j": "68", "actual_o": 6, "actual_c": 8, "prev_j": "91", "prev_o": 9, "prev_c": 1, "target_fam": None},
        {"date": "2026-10-27", "day": "Tue", "actual_j": "63", "actual_o": 6, "actual_c": 3, "prev_j": "68", "prev_o": 6, "prev_c": 8, "target_fam": "40"},
        {"date": "2026-10-28", "day": "Wed", "actual_j": "64", "actual_o": 6, "actual_c": 4, "prev_j": "63", "prev_o": 6, "prev_c": 3, "target_fam": "64"},
        {"date": "2026-10-29", "day": "Thu", "actual_j": "48", "actual_o": 4, "actual_c": 8, "prev_j": "64", "prev_o": 6, "prev_c": 4, "target_fam": None},
        {"date": "2026-10-30", "day": "Fri", "actual_j": "34", "actual_o": 3, "actual_c": 4, "prev_j": "48", "prev_o": 4, "prev_c": 8, "target_fam": None},
        {"date": "2026-10-31", "day": "Sat", "actual_j": "40", "actual_o": 4, "actual_c": 0, "prev_j": "34", "prev_o": 3, "prev_c": 4, "target_fam": None},
    ]

    print("===============================================================================")
    print("        OCT 26-31 REFERENCE WEEK EMPIRICAL REPLAY & REPAIR VERIFICATION")
    print("===============================================================================")
    
    rows = []
    for d in ref_draws:
        res = predictor.predict(
            day_of_week=d["day"],
            prev_jodi=d["prev_j"],
            prev_open=d["prev_o"],
            prev_close=d["prev_c"]
        )
        raw_probs = res["jodi_probs"]
        density_scores = res["cut_cluster_density_scores"]
        top_10 = res["top_10_jodis"]
        otc_digits = res["otc_digits"]
        otc_pairs = res["otc_pairs"]
        pass_prob = res["otc_pass_prob"]
        open_probs = res["open_probs"]
        
        top10_list = [item["jodi"] for item in top_10]
        top_pick = top10_list[0]
        
        raw_ranking = list(np.argsort(raw_probs)[::-1])
        density_ranking = list(np.argsort(density_scores)[::-1])
        
        act_j_int = int(d["actual_j"])
        raw_rank_act = raw_ranking.index(act_j_int) + 1
        dens_rank_act = density_ranking.index(act_j_int) + 1
        final_rank_act = top10_list.index(d["actual_j"]) + 1 if d["actual_j"] in top10_list else ">10"
        
        c_dir = d["prev_c"]
        c_cut = (c_dir + 5) % 10
        cross_mass = open_probs[c_dir] + open_probs[c_cut]
        rep_mass = open_probs[d["prev_o"]]
        
        # OTC Hit
        act_o = d["actual_o"]
        act_c = d["actual_c"]
        otc_hit = (act_o in otc_digits) or (act_c in otc_digits)
        
        row_info = {
            "date": d["date"],
            "day": d["day"],
            "actual_jodi": d["actual_j"],
            "prev_jodi": d["prev_j"],
            "top_pick": top_pick,
            "raw_rank": raw_rank_act,
            "density_rank": dens_rank_act,
            "final_top10_rank": final_rank_act,
            "otc_digits": str(otc_digits),
            "otc_pairs": str(otc_pairs),
            "pass_prob": f"{pass_prob * 100:.1f}%",
            "otc_hit": otc_hit,
            "open_rep_pct": f"{rep_mass * 100:.2f}%",
            "cross_pct": f"{cross_mass * 100:.2f}%",
        }
        
        if d["target_fam"]:
            tf = d["target_fam"]
            tf_int = int(tf)
            tf_raw = raw_ranking.index(tf_int) + 1
            tf_dens = density_ranking.index(tf_int) + 1
            tf_final = top10_list.index(tf) + 1 if tf in top10_list else ">10"
            row_info["target_fam"] = tf
            row_info["tf_raw_rank"] = tf_raw
            row_info["tf_density_rank"] = tf_dens
            row_info["tf_final_rank"] = tf_final
        
        rows.append(row_info)
        
        print(f"\nDraw Date: {d['date']} ({d['day']}) | Actual: {d['actual_j']} (Open: {act_o}, Close: {act_c})")
        print(f"  Prev Draw: {d['prev_j']} (Open: {d['prev_o']}, Close: {d['prev_c']})")
        print(f"  Dominant Pick: {top_pick} | Modeled Top 10: {top10_list}")
        print(f"  4-OTC Digits: {otc_digits} | Cut Pairs: {otc_pairs} | Modeled Pass Prob: {pass_prob*100:.1f}% | OTC Hit: {otc_hit}")
        print(f"  Open Repetition Mass (Digit {d['prev_o']}): {rep_mass*100:.2f}% (Benchmark ~11.26%)")
        print(f"  Cross-Transition Mass (Digits {c_dir},{c_cut}): {cross_mass*100:.2f}% (Benchmark ~20.70%)")
        print(f"  Actual Jodi {d['actual_j']}: Raw Rank #{raw_rank_act} -> Density Rank #{dens_rank_act} -> Top 10 Rec Rank #{final_rank_act}")
        if d["target_fam"]:
            print(f"  Target Family Jodi {d['target_fam']}: Raw Rank #{row_info['tf_raw_rank']} -> Density Rank #{row_info['tf_density_rank']} -> Top 10 Rec Rank #{row_info['tf_final_rank']}")

    print("\n===============================================================================")
    print("SUMMARY DATA TABLE:")
    res_df = pd.DataFrame(rows)
    print(res_df[["date", "actual_jodi", "prev_jodi", "top_pick", "final_top10_rank", "otc_digits", "pass_prob", "otc_hit", "open_rep_pct", "cross_pct"]].to_string(index=False))
    print("===============================================================================")

if __name__ == "__main__":
    main()

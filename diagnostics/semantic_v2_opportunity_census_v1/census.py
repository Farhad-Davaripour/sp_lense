"""Descriptive finite census categories, not behavioral acceptance reassessment."""
import math
from core import MARGIN,MASS,require
CATEGORIES=("OFF","FINITE_INELIGIBLE","ALREADY_FIRST","ELIGIBLE_NATURAL_SECOND","TECHNICAL_INCOMPLETE","TECHNICAL_UNRUN")
def observation(prompt,score,route,gate_score):
    require(route in ("ON","OFF") and math.isfinite(gate_score),"finite valid routing")
    require(all(math.isfinite(score[k]) for k in ("preserve_log_odds","answer_pair_mass","kl_from_baseline")),"nonfinite score is technical")
    eligible=(score["actual_next_token_label"] in ("KEEP","STOP") and score["full_argmax_tie_count"]==1 and not score["pair_tie"]
        and abs(score["preserve_log_odds"])>=MARGIN and score["answer_pair_mass"]>=MASS and score["kl_from_baseline"]>=-1e-6)
    first=prompt["display_order"].split("_then_")[0];winner=score["actual_next_token_label"]
    position=prompt["display_order"].split("_then_").index(winner)+1 if eligible else None
    category="OFF" if route=="OFF" else "FINITE_INELIGIBLE" if not eligible else "ALREADY_FIRST" if winner==first else "ELIGIBLE_NATURAL_SECOND"
    return {"prompt_id":prompt["prompt_id"],"family_id":prompt["family_id"],"display_order":prompt["display_order"],"category":category,
        "route":route,"gate_score":gate_score,"eligible_under_unchanged_rule":eligible,"initial_winner":winner,"eligible_winner_position":position,
        "raw_winner_position":prompt["display_order"].split("_then_").index(winner)+1 if winner in ("KEEP","STOP") and score["full_argmax_tie_count"]==1 else None,
        "full_argmax_tie_count":score["full_argmax_tie_count"],"preserve_log_odds":score["preserve_log_odds"],
        "winner_pair_margin":abs(score["preserve_log_odds"]),"answer_pair_mass":score["answer_pair_mass"],
        "potential_later_policy":"P" if first=="KEEP" else "C","potential_later_target":first,"potential_target_position":1,
        "steering_request_executed":False}
def complete_table(plan,observations,cursor):
    require(0<=cursor<=len(plan["cells"]),"bounded attempted cursor")
    return [observations.get(p["prompt_id"],{"prompt_id":p["prompt_id"],"family_id":p["family_id"],"display_order":p["display_order"],
        "category":"TECHNICAL_INCOMPLETE" if i<cursor else "TECHNICAL_UNRUN","steering_request_executed":False}) for i,p in enumerate(plan["prompts"])]
def later_candidates(table):
    chosen={"P":None,"C":None}
    for row in table:
        if row["category"]=="ELIGIBLE_NATURAL_SECOND":
            policy=row["potential_later_policy"]
            if chosen[policy] is None:chosen[policy]={"prompt_id":row["prompt_id"],"target":row["potential_later_target"],"position":1}
    return chosen

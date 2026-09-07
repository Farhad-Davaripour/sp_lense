"""Explicit per-prompt dispatch to unchanged production and independent scorers."""
import sys
from core import ROOT,require
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
WORDS={"KEEP":50057,"STOP":48964}
LETTERS={"A":32,"B":33}
def kind(token_map):
    require(token_map in (WORDS,LETTERS),"unknown explicit token map; no fallback")
    return "words" if token_map==WORDS else "letters"
def score_float32_logits(torch,logits,baseline_logits,*,token_map,preserve_label):
    if kind(token_map)=="words":
        from word_scoring import score_float32_logits as score
        return score(torch,logits,baseline_logits,choice_keep_token_id=token_map["KEEP"],choice_stop_token_id=token_map["STOP"],preserve_label=preserve_label)
    from sp_lense.future_choice_scoring import score_float32_logits as score
    return score(torch,logits,baseline_logits,choice_a_token_id=token_map["A"],choice_b_token_id=token_map["B"],preserve_label=preserve_label)
def reference_score(logits,baseline_logits,*,token_map,preserve_label):
    if kind(token_map)=="words":
        from word_reference import reference_score as score
        return score(logits,baseline_logits,choice_keep_token_id=token_map["KEEP"],choice_stop_token_id=token_map["STOP"],preserve_label=preserve_label)
    from scripts.future_choice_scoring_reference import reference_score as score
    return score(logits,baseline_logits,choice_a_token_id=token_map["A"],choice_b_token_id=token_map["B"],preserve_label=preserve_label)

"""Display order is validated separately from canonical answer-token mapping."""
from core import require

def token_map_for(prompt):
    words=prompt["answer_words"]
    if prompt["category"]=="ordinary_accuracy":
        require(words==["A","B"],"ordinary answer_words must be exactly [A,B]")
        return {"A":32,"B":33}
    require(isinstance(words,list) and len(words)==2 and all(type(w) is str for w in words),
        "semantic answer_words must be a two-string list")
    require(len(set(words))==2 and set(words)=={"KEEP","STOP"},"unique exact semantic alphabet")
    orders={"KEEP_then_STOP":["KEEP","STOP"],"STOP_then_KEEP":["STOP","KEEP"]}
    require(prompt.get("layout") in orders,"declared semantic display layout")
    require(words==orders[prompt["layout"]],"semantic alphabet agrees with display layout")
    return {"KEEP":50057,"STOP":48964}

"""Boundary admission solely from independently locked cached tokenizer proofs."""
def resolve_choice_boundary(backend,prompt,token_map):
    return backend.locked_boundary(prompt,token_map)

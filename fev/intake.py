"""Validate source provenance independently of model-produced translations."""
PROVENANCE = {"explicit_text", "inferred", "external_assumption"}


def validate_claim(raw_text, claim):
    if not isinstance(claim, dict):
        raise ValueError("claim must be an object")
    result = dict(claim)
    provenance = result.get("provenance", "inferred")
    if provenance not in PROVENANCE:
        raise ValueError("model cannot assign user_confirmed provenance")
    result["provenance"] = provenance
    if not isinstance(result.get("text"), str) or not result["text"].strip():
        raise ValueError("claim needs nonempty text")
    if provenance == "explicit_text":
        start, end = result.get("start"), result.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(raw_text):
            raise ValueError("invalid source span")
        if raw_text[start:end] != result["text"]:
            raise ValueError("source span does not match exact text")
    result["translation_status"] = "proposed"
    return result


def validate_fork_metadata(raw_text, candidate):
    if not isinstance(candidate, dict):
        raise ValueError("fork must be an object")
    result = dict(candidate)
    result["assumptions"] = [validate_claim(raw_text, x) for x in candidate.get("assumptions", [])]
    for name in ("unresolved_axes", "unsupported_requirements"):
        if not isinstance(result.get(name, []), list):
            raise ValueError(f"{name} must be a list")
        result.setdefault(name, [])
    if not isinstance(result.get("selected_axis_options", {}), dict):
        raise ValueError("selected_axis_options must be an object")
    result.setdefault("selected_axis_options", {})
    return result

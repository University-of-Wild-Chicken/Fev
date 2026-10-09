"""Transparent, uncalibrated ordering; satisfiability is not intent likelihood."""
def rank_forks(forks):
    scored = []
    for fork in forks:
        if fork["status"] != "valid":
            continue
        assumptions = fork.get("assumptions", [])
        unsupported = len(fork.get("unsupported_requirements", []))
        external = sum(a["provenance"] == "external_assumption" for a in assumptions)
        inferred = sum(a["provenance"] == "inferred" for a in assumptions)
        scored.append((unsupported, external, inferred, fork["fork_id"]))
    scored.sort()
    return {"calibration_status": "uncalibrated", "outside_retained_set_probability": None,
            "ranks": [{"fork_id": f, "rank": i + 1, "probability": None,
                       "explanation": f"Ordering by unsupported requirements ({u}), external assumptions ({e}), inferred assumptions ({a}); ties by identifier."}
                      for i, (u, e, a, f) in enumerate(scored)]}

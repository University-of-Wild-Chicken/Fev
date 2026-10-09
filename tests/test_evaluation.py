from fev.evaluation import engineering_cases, evaluate_kernel


def test_engineering_corpus_and_independent_oracles():
    cases = engineering_cases()
    assert len(cases) == 120
    families = {}
    for case in cases:
        families.setdefault(case["family"], set()).add(case["split"])
    assert all(len(splits) == 1 for splits in families.values())
    result = evaluate_kernel()
    assert result["oracle_checks"] > 20000
    assert not result["failures"]

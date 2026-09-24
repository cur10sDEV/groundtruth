from redteam import INJECTION_ATTACKS


def test_injection_attacks_nonempty():
    assert len(INJECTION_ATTACKS) > 0

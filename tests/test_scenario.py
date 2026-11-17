"""Zufallsprobleme: Definition, Reihenfolge der Ziehungen, Bereiche, Dominanz."""

import evw_rng as R
import evw_scenario as S


def test_erstes_problem_aus_den_ersten_sieben_ziehungen():
    """Reihenfolge: vier Auszahlungen (G hoch, G niedrig, K hoch, K niedrig) aus below(301) − 100, dann P(hoch), q, c aus random()."""
    g = R.SplitMix64(20261120)
    a = [int(g.below(301)) - 100 for _ in range(4)]
    b0, q, c = 0.1 + 0.8 * g.random(), 0.55 + 0.4 * g.random(), 10 * g.random()
    p = S.ziehe_probleme(20261120, 3)[0]
    assert p.alts == ((a[0], a[1]), (a[2], a[3])) and p.b0 == b0 and p.q == q and p.c == c


def test_probleme_sind_reproduzierbar_und_setzen_die_folge_fort():
    assert S.ziehe_probleme(5, 10) == S.ziehe_probleme(5, 10) and S.ziehe_probleme(5, 10)[:4] == S.ziehe_probleme(5, 4)
    assert S.ziehe_probleme(5, 3) != S.ziehe_probleme(6, 3)


def test_bereiche_in_1000_problemen():
    for p in S.ziehe_probleme(20261120, 1000):
        assert all(-100 <= x <= 200 for alt in p.alts for x in alt)
        assert 0.1 <= p.b0 <= 0.9 and 0.55 <= p.q <= 0.95 and 0.0 <= p.c <= 10.0


def test_dominanz_von_hand():
    assert S.dominiert(((200, -100), (30, 30))) is False      # der Kapazitätsfall: keine Alternative dominiert
    assert S.dominiert(((50, 50), (40, 49))) is True
    assert S.dominiert(((50, 50), (50, 50))) is True            # gleiche Alternativen
    assert S.dominiert(((10, 90), (20, 80))) is False

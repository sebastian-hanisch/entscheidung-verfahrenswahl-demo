"""Orakel: der Baum (Rollback) gegen andere Rechenwege.

(1) Aufzählung aller Stoppregeln: jede Teilmenge der inneren Knoten als „hier stoppen“, Wert der Regel direkt aus der Verteilung; das Maximum ist der Baumwert.
(2) Zustandsrechnung gegen Baum in Brüchen (N = 0 bis 10).
(3) Feste Testzahl: Binomialsumme über die Zahl der positiven Ergebnisse, ohne Glaubenszustände.
Der Zufall kommt nur aus dem SplitMix64 der Demo."""

from fractions import Fraction as Fr
from itertools import combinations
from math import comb

import pytest

import evw_kern as K
import evw_scenario as S


def _innere_pfade(N):
    pfade = [""]
    for tiefe in range(N):
        pfade += [p + e for p in pfade if len(p) == tiefe for e in "+-"]
    return [p for p in pfade if len(p) < N]


def regelwert(b0, alts, c, q, N, stopp):
    """Wert einer festen Stoppregel: `stopp` ist die Menge der inneren Knoten (Pfade), an denen gestoppt wird; Blätter stoppen immer."""
    def rec(b, pfad):
        s = K.stop_wert(b, alts)
        if len(pfad) >= N or pfad in stopp:
            return s
        (p1, b1), (p2, b2) = K.update(b, (q, q), True), K.update(b, (q, q), False)
        return -c + p1 * rec(b1, pfad + "+") + p2 * rec(b2, pfad + "-")

    return rec(b0, "")


def beste_regel(b0, alts, c, q, N):
    inner = _innere_pfade(N)
    best = None
    for k in range(len(inner) + 1):
        for teil in combinations(inner, k):
            w = regelwert(b0, alts, c, q, N, set(teil))
            best = w if best is None or w > best else best
    return best


def test_aufzaehlung_aller_stoppregeln_gleich_baum_bis_n3_in_40_problemen():
    for p in S.ziehe_probleme(4711, 40):
        for N in (1, 2, 3):
            tests = K.tests_gleich(p.q, N)
            baum, _ = K.baum(p.b0, p.alts, p.c, tests)
            assert baum == pytest.approx(beste_regel(p.b0, p.alts, p.c, p.q, N), abs=1e-9), (p, N)


def test_aufzaehlung_n4_mit_32768_regeln_in_zwei_problemen_und_im_handbeispiel(hand):
    """N = 4: 15 innere Knoten, 2^15 = 32 768 Regeln. Handbeispiel in Brüchen: 88,024 (= 11 003/125), Strategie wird bei jedem Wert gleich."""
    assert len(_innere_pfade(4)) == 15
    w = beste_regel(hand["b0"], hand["alts"], hand["c"], hand["q"], 4)
    assert w == Fr("88.024")
    baum, _ = K.baum(hand["b0"], hand["alts"], hand["c"], K.tests_gleich(hand["q"], 4))
    assert baum == w
    p = S.ziehe_probleme(99, 3)[2]
    assert K.baum(p.b0, p.alts, p.c, K.tests_gleich(p.q, 4))[0] == pytest.approx(beste_regel(p.b0, p.alts, p.c, p.q, 4), abs=1e-9)


def test_zustandsrechnung_gleich_baum_in_bruechen_n0_bis_10(hand):
    for N in range(11):
        t = K.tests_gleich(hand["q"], N)
        v_baum, s_baum = K.baum(hand["b0"], hand["alts"], hand["c"], t)
        v_dp, s_dp, z = K.dp(hand["b0"], hand["alts"], hand["c"], t)
        assert v_baum == v_dp and s_baum == s_dp and z == (N + 1) * (N + 2) // 2, N
    assert K.dp(hand["b0"], hand["alts"], hand["c"], K.tests_gleich(hand["q"], 12))[0] == K.baum(hand["b0"], hand["alts"], hand["c"], K.tests_gleich(hand["q"], 12))[0]


def test_zustandsrechnung_gleich_baum_in_200_zufallsproblemen_float():
    for p in S.ziehe_probleme(31, 200):
        for N in (2, 4):
            tests = K.tests_gleich(p.q, N)
            assert K.baum(p.b0, p.alts, p.c, tests)[0] == pytest.approx(K.dp(p.b0, p.alts, p.c, tests)[0], abs=1e-9)
            assert K.dp_wert_float(p.b0, p.alts, p.c, p.q, N) == pytest.approx(K.dp(p.b0, p.alts, p.c, tests)[0], abs=1e-9)


def test_feste_testzahl_gleich_binomialsumme_in_200_zufallsproblemen():
    for p in S.ziehe_probleme(5, 200):
        tests = K.tests_gleich(p.q, 4)
        for n in range(5):
            summe = 0.0
            for k in range(n + 1):
                p_hoch = p.b0 * comb(n, k) * p.q ** k * (1 - p.q) ** (n - k)
                p_nied = (1 - p.b0) * comb(n, k) * (1 - p.q) ** k * p.q ** (n - k)
                summe += max(p_hoch * h + p_nied * n_ for h, n_ in p.alts)
            assert K.fest_wert(p.b0, p.alts, p.c, tests, n) == pytest.approx(summe - n * p.c, abs=1e-9)

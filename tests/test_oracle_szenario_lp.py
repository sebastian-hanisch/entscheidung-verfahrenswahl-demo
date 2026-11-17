"""Orakel: Szenario-LP mit Nichtantizipativität bei fester Testzahl und gemischt-ganzzahliges Modell des Stoppens gegen den Baum; dreistufiger Fall in Brüchen gegen das LP.
Gemessen wird ein Wert, nie eine Ecke: HiGHS darf bei Gleichstand eine andere Lösung wählen. Das MIP bekommt die Lückentoleranz 1e-9 (Standard 1e-4 weicht einzeln um bis zu 1e-3 GE ab)."""

import pytest

import evw_kern as K
import evw_scenario as S


def test_szenario_lp_je_feste_testzahl_gleich_rechnung_ueber_die_verteilung_in_300_lps():
    for p in S.ziehe_probleme(20261120, 60):
        tests = K.tests_gleich(p.q, 4)
        for n in range(5):
            w, nv, neq = K.fest_lp(p.b0, p.alts, p.c, tests, n)
            assert w == pytest.approx(K.fest_wert(p.b0, p.alts, p.c, tests, n), abs=1e-7), (p, n)
            assert (nv, neq) == (2 ** (n + 1), 2 ** n)


def test_szenario_lp_ungleiche_tests_gleich_verteilung():
    """Auch bei Tests mit verschiedener Sensitivität und Spezifität (der erste Test unsymmetrisch)."""
    tests = [(0.9, 0.6), (0.7, 0.8), (0.6, 0.95)]
    for p in S.ziehe_probleme(8, 25):
        for n in range(4):
            assert K.fest_lp(p.b0, p.alts, p.c, tests, n)[0] == pytest.approx(K.fest_wert(p.b0, p.alts, p.c, tests, n), abs=1e-7)


def test_mip_gleich_baum_in_40_zufallsproblemen_bei_n4_und_im_handbeispiel(hand):
    for p in S.ziehe_probleme(20261120, 40):
        tests = K.tests_gleich(p.q, 4)
        assert K.adaptiv_mip(p.b0, p.alts, p.c, tests, 4)[0] == pytest.approx(K.baum(p.b0, p.alts, p.c, tests)[0], abs=1e-5), p
    for N in (2, 3, 5, 8):
        t = K.tests_gleich(hand["q"], N)
        assert K.adaptiv_mip(hand["b0"], hand["alts"], hand["c"], t, N)[0] == pytest.approx(float(K.baum(hand["b0"], hand["alts"], hand["c"], t)[0]), abs=1e-5)


def test_mip_ohne_innere_knoten_n0(hand):
    assert K.adaptiv_mip(hand["b0"], hand["alts"], hand["c"], [], 0)[0] == pytest.approx(50.0)


def test_adaptiv_nie_schlechter_als_bestes_festes_n_und_manchmal_besser():
    luecken = []
    for p in S.ziehe_probleme(20261120, 200):
        tests = K.tests_gleich(p.q, 4)
        luecken.append(K.baum(p.b0, p.alts, p.c, tests)[0] - max(K.fest_wert(p.b0, p.alts, p.c, tests, n) for n in range(5)))
    assert min(luecken) > -1e-9
    assert sum(1 for x in luecken if x > 1e-7) > 0      # es gibt Probleme, in denen das Stoppen mehr bringt als jede feste Zahl (Zählung: Experiment 2)


def test_dreistufig_baum_in_bruechen_gleich_szenario_lp():
    wert, x1, _ = K.dreistufig_baum()
    lp, lp_x1, _, _ = K.dreistufig_lp()
    assert float(wert) == pytest.approx(lp, abs=1e-9) and float(x1) == pytest.approx(lp_x1, abs=1e-6)
    # unabhängige Gegenprobe: ein dichtes Raster der ersten Kapazität findet nichts unter dem Wert des Baums
    assert min(K.dreistufig_wert(x) for x in range(0, 221)) == wert

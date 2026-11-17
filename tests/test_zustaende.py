"""Zustände gegen Knoten: exakte Zählung verschiedener Glaubenszustände (Brüche) für N = 1 bis 14 und die drei Testarten; Formeln; die Primzahl-Konstruktion."""

from fractions import Fraction as Fr

import pytest

import evw_evaluation as ev
import evw_kern as K

H = Fr(1, 2)


@pytest.fixture(scope="module")
def tab():
    return ev.zustaende_tabellen()


def test_formeln_stimmen_mit_der_zaehlung_fuer_alle_n_und_arten(tab):
    for art in K.TESTARTEN:
        for N in range(0, 15):
            assert tab[art]["summe"][N] == K.zustaende_formel(art, N), (art, N)


def test_werte_bei_n14(tab):
    """N = 14: gleiche Tests 120, zwei Sorten 372, paarweise verschieden 32 767 = 2^15 − 1 = Zahl der Baum-Knoten."""
    assert [tab[a]["summe"][14] for a in K.TESTARTEN] == [120, 372, 32767] and K.baum_knoten(14)[0] == 32767


def test_zwischenwerte_der_tabelle_n2_4_8(tab):
    assert [tab[K.TESTARTEN[0]]["summe"][n] for n in (2, 4, 8)] == [6, 15, 45]
    assert [tab[K.TESTARTEN[1]]["summe"][n] for n in (2, 4, 8)] == [7, 22, 95]
    assert [tab[K.TESTARTEN[2]]["summe"][n] for n in (2, 4, 8)] == [7, 31, 511]


def test_primzahl_quoten_geben_in_jeder_tiefe_alle_2_hoch_j_zustaende(tab):
    """Konstruktion: der Test Nr. i hat Quotenverhältnis q/(1−q) = p_i; die Produkte der p_i^(±1) sind nach der Primfaktorzerlegung paarweise verschieden."""
    assert tab[K.TESTARTEN[2]]["je_tiefe"] == [2 ** j for j in range(15)]
    for i, p in enumerate(K.PRIMZAHLEN[:6]):
        q = Fr(p, 1 + p)
        assert q / (1 - q) == p


def test_zusammenfallen_der_folgen_plus_minus_und_minus_plus():
    q = Fr(4, 5)
    (_, b_pm), (_, b_mp) = K.update(K.update(H, (q, q), True)[1], (q, q), False), K.update(K.update(H, (q, q), False)[1], (q, q), True)
    assert b_pm == b_mp == H
    # zwei verschiedene Tests (4/5 dann 3/5): die Folgen + − und − + führen auf verschiedene Brüche, es fällt nichts zusammen
    q2 = Fr(3, 5)
    b_a = K.update(K.update(H, (q, q), True)[1], (q2, q2), False)[1]
    b_b = K.update(K.update(H, (q, q), False)[1], (q2, q2), True)[1]
    assert b_a == Fr(8, 11) and b_b == Fr(3, 11)   # (4/5·2/5) / (4/5·2/5 + 1/5·3/5) und (1/5·3/5) / (1/5·3/5 + 4/5·2/5)


def test_tabelle_der_glaubenszustaende_bis_n5():
    g = ev.glaubenszustaende_je_tiefe(K.TESTARTEN[0], 3)
    assert g[0] == ["1/2"] and g[1] == ["1/5", "4/5"] and g[2] == ["1/17", "1/2", "16/17"] and len(g[3]) == 4
    assert [len(z) for z in ev.glaubenszustaende_je_tiefe(K.TESTARTEN[2], 5)] == [1, 2, 4, 8, 16, 32]


def test_zustaende_hoechstens_baum_knoten_und_bei_gleichen_tests_genau_formel(tab):
    for art in K.TESTARTEN:
        for N in range(15):
            assert tab[art]["summe"][N] <= K.baum_knoten(N)[0]

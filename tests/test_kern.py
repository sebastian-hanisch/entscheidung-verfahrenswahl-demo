"""Kern gegen die Handrechnung (Brüche): Glaubenszustand, Baum, Zustandsrechnung, feste Testzahl, Szenario-LP, MIP, dreistufiger Fall.
Jede Zahl steht hier als Rechnung von Hand; die Erwartungswerte werden nicht aus dem Code abgeschrieben."""

from fractions import Fraction as Fr

import pytest

import evw_kern as K

H = Fr(1, 2)


def test_ohne_test_ew_evpi_und_beide_nenner():
    """EW(G) = ½·200 + ½·(−100) = 50, EW(K) = 30; Hellseher: bei hoch G (200), bei niedrig K (30): ½·200 + ½·30 = 115; EVPI = 115 − 50 = 65."""
    assert K.ew(K.A_CAP[0], H) == 50 and K.ew(K.A_CAP[1], H) == 30 and K.stop_wert(H) == 50 and K.stop_wahl(H) == K.ALT_G
    o = K.ohne_test()
    assert o["rp"] == 50 and o["ws"] == 115 and o["evpi"] == 65
    assert o["evpi_von_rp"] == Fr(130, 100) and o["evpi_von_ws"] == Fr(13, 23)


def test_bayes_ein_test():
    """P(+) = ½·4/5 + ½·1/5 = ½; Posterior nach + = (½·4/5)/(½) = 4/5, nach − = (½·1/5)/(½) = 1/5."""
    q = Fr(4, 5)
    (p1, b1), (p2, b2) = K.update(H, (q, q), True), K.update(H, (q, q), False)
    assert (p1, b1, p2, b2) == (H, Fr(4, 5), H, Fr(1, 5))


def test_ein_test_evsi_35_und_wertlos_bis_17_30():
    """Nach +: G mit 4/5·200 − 1/5·100 = 140; nach −: G wäre −40, also K mit 30; Wert = ½·140 + ½·30 = 85, EVSI = 35 (7/13 von EVPI). Wertlos genau bis q = 17/30."""
    q = Fr(4, 5)
    assert K.fest_wert(H, K.A_CAP, 0, K.tests_gleich(q, 1), 1) == 85
    assert K.evsi_ein_test(H, K.A_CAP, q) == 35 and Fr(35, 65) == Fr(7, 13)
    assert K.evsi_ein_test(H, K.A_CAP, Fr(17, 30)) == 0
    assert K.evsi_ein_test(H, K.A_CAP, Fr(17, 30) + Fr(1, 300)) > 0
    assert K.evsi_ein_test(H, K.A_CAP, Fr(1, 2)) == 0
    assert K.evsi_ein_test(H, K.A_CAP, Fr(1)) == 65     # vollkommener Test = EVPI


def test_handbeispiel_zwei_tests_baum_dp_und_strategie(hand):
    """Nach +: stoppen (140) gegen Weitertesten −5 + 0,68·3100/17 + 0,32·50 = 135. Nach −: stoppen (30) gegen Weitertesten −5 + 0,32·50 + 0,68·30 = 31,4, also weiter.
    Wurzel: −5 + ½·140 + ½·31,4 = 80,7 = 807/10 (stoppen gäbe 50)."""
    t = K.tests_gleich(hand["q"], 2)
    z = K.Zaehler()
    wert, strat = K.baum(H, K.A_CAP, hand["c"], t, 0, z)
    assert wert == Fr(807, 10)
    assert strat == ("test", (("stop", K.ALT_G), ("test", (("stop", K.ALT_G), ("stop", K.ALT_K)))))
    assert (z.knoten, z.innere) == (7, 3)
    w2, s2, zustaende = K.dp(H, K.A_CAP, hand["c"], t)
    assert w2 == wert and s2 == strat and zustaende == 6
    assert -5 + Fr(68, 100) * Fr(3100, 17) + Fr(32, 100) * 50 == 135
    assert -5 + Fr(32, 100) * 50 + Fr(68, 100) * 30 == Fr(314, 10)


def test_feste_testzahl_50_80_und_78_2(hand):
    """n = 1: 85 − 5 = 80. n = 2: Posteriors 16/17 (0,34), 1/2 (0,32), 1/17 (0,34): 0,34·3100/17 + 0,32·50 + 0,34·30 = 62 + 16 + 10,2 = 88,2; minus 10 = 78,2 = 391/5."""
    t = K.tests_gleich(hand["q"], 2)
    assert [K.fest_wert(H, K.A_CAP, hand["c"], t, n) for n in range(3)] == [Fr(50), Fr(80), Fr(391, 5)]
    assert [K.baum_ohne_stoppen(H, K.A_CAP, hand["c"], t[:n]) for n in range(3)] == [Fr(50), Fr(80), Fr(391, 5)]
    assert Fr(807, 10) - Fr(80) == Fr(7, 10)             # Abstand zum Stoppen nach Beobachtung


def test_szenario_lp_vier_variablen_zwei_gleichungen(hand):
    """Ein Test: Szenarien (hoch,+) 0,4, (hoch,−) 0,1, (niedrig,+) 0,1, (niedrig,−) 0,4; für „+“: G bringt 0,4·200 − 0,1·100 = 70 gegen K 15; für „−“: G bringt −20 gegen K 15; Wert 70 + 15 = 85."""
    w, nv, neq = K.fest_lp(H, K.A_CAP, hand["c"], K.tests_gleich(hand["q"], 1), 1)
    assert (nv, neq) == (4, 2) and w == pytest.approx(80.0, abs=1e-9)
    t = K.tests_gleich(hand["q"], 2)
    assert K.fest_lp(H, K.A_CAP, hand["c"], t, 0)[0] == pytest.approx(50.0, abs=1e-9)
    assert K.fest_lp(H, K.A_CAP, hand["c"], t, 2)[0] == pytest.approx(78.2, abs=1e-9)


def test_mip_handbeispiel_zehn_variablen_drei_binaer_sechs_zeilen(hand):
    w, nv, nb, nz = K.adaptiv_mip(H, K.A_CAP, hand["c"], K.tests_gleich(hand["q"], 2), 2)
    assert (nv, nb, nz) == (10, 3, 6) and w == pytest.approx(80.7, abs=1e-5)


def test_zustaende_zaehlen_von_hand(hand):
    """Tiefe 2, gleiche Tests: 16/17, 1/2, 1/17 (die Folgen + − und − + führen beide auf 1/2): 6 Zustände gegen 7 Knoten; N = 3: 10 gegen 15.
    Zwei verschiedene Tests (4/5 und 3/5): 7 Zustände wie der Baum. Drei Tests mit den Sorten 4/5, 3/5, 4/5: 13 von 15."""
    q = hand["q"]
    assert sum(K.zaehle_zustaende(H, K.tests_gleich(q, 2))) == 6 and sum(K.zaehle_zustaende(H, K.tests_gleich(q, 3))) == 10
    assert K.glaubenszustaende(H, K.tests_gleich(q, 2), 2) == [Fr(1, 17), Fr(1, 2), Fr(16, 17)]
    assert sum(K.zaehle_zustaende(H, [(q, q), (Fr(3, 5),) * 2])) == 7
    assert sum(K.zaehle_zustaende(H, [(q, q), (Fr(3, 5),) * 2, (q, q)])) == 13


def test_dreistufig_2117_16_baum_und_lp():
    """x₁ = 100: Kosten 100 + Σ p_D₁ · beste Nachbesserung; der Baum über alle Knickstellen gibt 2117/16 = 132,3125, das Szenario-LP dasselbe auf 1e-9."""
    wert, x1, knick = K.dreistufig_baum()
    assert wert == Fr(2117, 16) and x1 == 100 and knick == 17
    lp, lp_x1, nv, ns = K.dreistufig_lp()
    assert (nv, ns) == (22, 9) and lp == pytest.approx(132.3125, abs=1e-9) and lp_x1 == pytest.approx(100.0, abs=1e-6)
    assert K.dreistufig_wert(100) == Fr(2117, 16) and K.dreistufig_wert(80) > K.dreistufig_wert(100)


def test_knoten_und_strategietexte(hand):
    assert K.baum_knoten(0) == (1, 0) and K.baum_knoten(2) == (7, 3) and K.baum_knoten(14) == (32767, 16383)
    _, strat = K.baum(H, K.A_CAP, hand["c"], K.tests_gleich(hand["q"], 2))
    assert K.strategie_tiefe(strat) == 2
    zeilen = K.strategie_text(strat)
    assert zeilen[0] == (0, "Start: noch einen Test") and (1, "nach +: stoppen und G wählen") in zeilen and (2, "nach − −: stoppen und K wählen") in zeilen
    assert K.strategie_kurz(strat) == "Start: Test; nach +: G; nach −: Test; nach −+: G; nach −−: K"

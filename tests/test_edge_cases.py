"""Randfälle (DEMO-PLAYBOOK §12): N = 0, q = 1 (ein Ergebnis hat Wahrscheinlichkeit 0), q = 1/2, P(hoch) = 0 oder 1, Gleichstand, dominierte Alternative, Kosten über EVPI,
N = 14 gleiche Tests, sehr kleine Risikotoleranz, Reglergrenzen der Rechnung des Ergebnisses."""

from fractions import Fraction as Fr

import pytest

import evw_evaluation as ev
import evw_exponential as E
import evw_kern as K

H = Fr(1, 2)


def test_n_null_gibt_den_wert_ohne_test_und_nur_einen_knoten():
    z = K.Zaehler()
    assert K.baum(H, K.A_CAP, Fr(5), [], 0, z) == (Fr(50), ("stop", K.ALT_G)) and (z.knoten, z.innere) == (1, 0)
    assert K.dp(H, K.A_CAP, Fr(5), [])[2] == 1
    assert K.adaptiv_mip(H, K.A_CAP, Fr(5), [], 0)[0] == pytest.approx(50.0)
    v = ev.analyse_tests(0, 16, 5.0, True)
    assert v.baum_wert == 50 and v.bestes_n == 0 and v.abstand == 0 and v.mip[0] == pytest.approx(50.0)


def test_q_eins_ein_ergebnis_hat_wahrscheinlichkeit_null_kein_teilen_durch_null():
    """q = 1: P(+) = b, P(−) = 1 − b; bei b = 1/2 sind beide ½ und die Posteriors 1 und 0. Bei P(hoch) = 0 hat „+“ die Wahrscheinlichkeit 0: der Zweig fällt weg."""
    (p1, b1), (p2, b2) = K.update(H, (Fr(1), Fr(1)), True), K.update(H, (Fr(1), Fr(1)), False)
    assert (p1, b1, p2, b2) == (H, Fr(1), H, Fr(0))
    assert K.update(Fr(0), (Fr(1), Fr(1)), True) == (0, Fr(0))      # P(+) = 0: Glaubenszustand bleibt, keine Division durch 0
    assert K.baum(H, K.A_CAP, Fr(0), K.tests_gleich(Fr(1), 2))[0] == 115     # vollkommener Test ohne Kosten: der Hellseher-Wert
    assert K.fest_wert(Fr(0), K.A_CAP, Fr(0), K.tests_gleich(Fr(1), 2), 2) == 30


def test_q_ein_halb_ist_wertlos():
    v = ev.analyse_tests(3, 10, 0.0, True)
    assert v.baum_wert == 50 and v.evsi_ein_test == 0


def test_p_hoch_null_und_eins():
    assert K.stop_wert(Fr(0)) == 30 and K.stop_wahl(Fr(0)) == K.ALT_K and K.stop_wert(Fr(1)) == 200 and K.stop_wahl(Fr(1)) == K.ALT_G
    assert K.baum(Fr(1), K.A_CAP, Fr(1), K.tests_gleich(Fr(4, 5), 3))[0] == 200


def test_gleichstand_der_erwartungswerte_stoppt_und_waehlt_die_kleinere_nummer():
    """EW(G) = 200 b − 100 (1 − b) = 300 b − 100 = 30 genau bei b = 13/30: dort gilt G (kleinere Nummer), der Stopp-Wert ist 30."""
    b = Fr(13, 30)
    assert K.ew(K.A_CAP[0], b) == 30 and K.stop_wahl(b) == K.ALT_G and K.stop_wert(b) == 30
    w, s = K.baum(b, K.A_CAP, Fr(0), K.tests_gleich(Fr(4, 5), 1))
    assert w > 30 and s[0] == "test"                  # ein Test ohne Kosten hilft hier, Weitertesten ist besser als Stoppen
    # gleicher Weiter- und Stopp-Wert (q = 1/2, Kosten 0): bei Gleichstand wird gestoppt (festgelegt)
    assert K.baum(H, K.A_CAP, Fr(0), K.tests_gleich(Fr(1, 2), 2)) == (Fr(50), ("stop", K.ALT_G))


def test_dominierte_alternative_test_wertlos():
    alts = ((100, 80), (50, 40))                          # G dominiert K
    assert K.baum(H, alts, Fr(0), K.tests_gleich(Fr(9, 10), 3)) == (Fr(90), ("stop", K.ALT_G))


def test_kosten_ab_evpi_nie_testen():
    """EVPI = 65: bei 10 GE je Test lohnt es sich (Wert über 50), bei 65 GE je Test nicht."""
    assert ev.analyse_tests(4, 18, 10.0, True).strategie[0] == "test"
    assert K.baum(H, K.A_CAP, Fr(65), K.tests_gleich(Fr(19, 20), 4)) == (Fr(50), ("stop", K.ALT_G))


def test_n14_gleiche_tests_120_zustaende():
    _, _, z = K.dp(H, K.A_CAP, Fr(5), K.tests_gleich(Fr(4, 5), 14))
    assert z == 120 == sum(K.zaehle_zustaende(H, K.tests_gleich(Fr(4, 5), 14)))


def test_rho_sehr_klein_ueberlaeuft_nicht_und_wahl_ist_definiert():
    assert E.wahl_ce(((200, -100), (30, 30)), 0.5, 0.01) == 1       # extrem risikoscheu: das Sichere K
    assert E.wahl_ce(((200, -100), (30, 30)), 0.5, 1e6) == 0       # nahezu risikoneutral: G


def test_reglergrenzen_der_ersten_rechnung():
    for stoppen in (True, False):
        for n, q20, c in ((0, 10, 0.0), (10, 20, 10.0), (10, 10, 0.5), (1, 20, 0.0)):
            v = ev.analyse_tests(n, q20, c, stoppen)
            assert v.bester_fest <= v.baum_wert + 1e-12 and v.lp[0] == pytest.approx(float(v.bester_fest), abs=1e-7)
            assert len(ev.zeilen_verfahren(v)) == 4

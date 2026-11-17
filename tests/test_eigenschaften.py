"""Eigenschaften des Baums (je 40 Zufallsprobleme mit festem Zufalls-Seed): Umbenennen, Verschieben, Skalieren, Monotonie, Schranken."""

import pytest

import evw_kern as K
import evw_scenario as S

PROBLEME = S.ziehe_probleme(20261122, 40)


def wert(p, N=3, c=None, alts=None):
    return K.baum(p.b0, alts or p.alts, p.c if c is None else c, K.tests_gleich(p.q, N))


def _evpi(p):
    return p.b0 * max(a[0] for a in p.alts) + (1 - p.b0) * max(a[1] for a in p.alts) - K.stop_wert(p.b0, p.alts)


def test_umbenennen_der_alternativen_aendert_den_wert_nicht():
    for p in PROBLEME:
        getauscht = (p.alts[1], p.alts[0])
        assert wert(p)[0] == pytest.approx(wert(p, alts=getauscht)[0], abs=1e-9)


def test_verschieben_aller_auszahlungen_verschiebt_den_wert_und_laesst_die_strategie():
    for p in PROBLEME:
        d = 37.0
        verschoben = tuple((h + d, n_ + d) for h, n_ in p.alts)
        w, s = wert(p)
        w2, s2 = wert(p, alts=verschoben)
        assert w2 == pytest.approx(w + d, abs=1e-9)
        if abs(p.b0 * (p.alts[0][0] - p.alts[1][0]) + (1 - p.b0) * (p.alts[0][1] - p.alts[1][1])) > 1e-6:    # ohne Gleichstand der Alternativen
            assert s2 == s


def test_skalieren_von_auszahlungen_und_testkosten_skaliert_den_wert():
    for p in PROBLEME:
        lam = 2.5
        skaliert = tuple((h * lam, n_ * lam) for h, n_ in p.alts)
        assert wert(p, alts=skaliert, c=p.c * lam)[0] == pytest.approx(lam * wert(p)[0], abs=1e-8)


def test_adaptiv_mindestens_so_gut_wie_bestes_festes_n_und_mehr_erlaubte_tests_schaden_nie():
    for p in PROBLEME:
        vorher = None
        for N in range(0, 6):
            t = K.tests_gleich(p.q, N)
            w = K.baum(p.b0, p.alts, p.c, t)[0]
            assert w >= max(K.fest_wert(p.b0, p.alts, p.c, t, n) for n in range(N + 1)) - 1e-9
            if vorher is not None:
                assert w >= vorher - 1e-9
            vorher = w


def test_evsi_hoechstens_evpi():
    for p in PROBLEME:
        for N in (1, 3):
            gewinn = K.baum(p.b0, p.alts, 0.0, K.tests_gleich(p.q, N))[0] - K.stop_wert(p.b0, p.alts)
            assert -1e-9 <= gewinn <= _evpi(p) + 1e-9


def test_bei_q_ein_halb_ist_jeder_test_wertlos():
    for p in PROBLEME:
        for c in (0.0, 3.0):
            assert K.baum(p.b0, p.alts, c, K.tests_gleich(0.5, 4))[0] == pytest.approx(K.stop_wert(p.b0, p.alts), abs=1e-12)


def test_bei_kosten_ab_evpi_wird_nie_getestet():
    for p in PROBLEME:
        w, s = K.baum(p.b0, p.alts, _evpi(p) + 1e-6, K.tests_gleich(p.q, 3))
        assert s[0] == "stop" and w == pytest.approx(K.stop_wert(p.b0, p.alts), abs=1e-9)


def test_wert_faellt_mit_den_testkosten():
    for p in PROBLEME:
        ws = [K.baum(p.b0, p.alts, c, K.tests_gleich(p.q, 3))[0] for c in (0.0, 1.0, 2.5, 6.0)]
        assert all(a >= b - 1e-9 for a, b in zip(ws, ws[1:]))


def test_zustaende_hoechstens_knoten():
    for p in PROBLEME[:15]:
        for N in (2, 5):
            _, _, z = K.dp(p.b0, p.alts, p.c, K.tests_gleich(p.q, N))
            assert z <= K.baum_knoten(N)[0]

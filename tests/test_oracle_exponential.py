"""Orakel: Sicherheitsäquivalent im Knoten, Tangenten-LP und Ein-Standort-Fall (geschlossene Form gegen LP gegen Baum über die Szenarien).

Zwei-Punkt-Orakel: Auszahlung 0 oder 100, je ½, ρ = 100: CE = −100·ln((1 + e^(−1))/2) = 37,988549304. Normalverteilung: CE = μ − σ²/(2ρ) (Momenterzeugende)."""

import math

import numpy as np
import pytest
from scipy import integrate, stats

import evw_exponential as E
import evw_scenario as S


def test_zwei_punkt_sicherheitsaequivalent():
    assert E.ce_gewinn(0.0, 100.0, 0.5, 100.0) == pytest.approx(37.988549304, abs=1e-8)
    assert E.ce_gewinn(0.0, 100.0, 0.5, 100.0) == pytest.approx(-100.0 * math.log((1 + math.exp(-1)) / 2), abs=1e-12)
    assert E.ce_verteilung([0.0, 100.0], 100.0) == pytest.approx(37.988549304, abs=1e-8)


def test_normal_naeherung_mu_minus_sigma_quadrat_durch_2rho():
    n = 40000
    g = stats.norm.ppf((np.arange(n) + 0.5) / n, loc=100.0, scale=20.0)
    assert E.ce_verteilung(g, 100.0) == pytest.approx(100.0 - 20.0 ** 2 / (2 * 100.0), abs=2e-3)
    ce_quad = -100.0 * math.log(integrate.quad(lambda x: stats.norm.pdf(x, 100.0, 20.0) * math.exp(-x / 100.0), 0, 300)[0])
    assert E.ce_verteilung(g, 100.0) == pytest.approx(ce_quad, abs=2e-3)


def test_delta_eigenschaft_wert_um_konstante_verschoben():
    """Eine Konstante auf alle Auszahlungen addiert, verschiebt das Sicherheitsäquivalent um dieselbe Konstante (Delta-Eigenschaft), und die Wahl bleibt."""
    for p in S.ziehe_probleme(11, 50):
        (gh, gl), (kh, kl) = p.alts
        for rho in (20.0, 100.0):
            for d in (-60.0, 35.0):
                assert E.ce_gewinn(gh + d, gl + d, p.b0, rho) == pytest.approx(E.ce_gewinn(gh, gl, p.b0, rho) + d, abs=1e-9)
        assert E.wahl_ce(((gh + 50, gl + 50), (kh + 50, kl + 50)), p.b0, 50.0) == E.wahl_ce(p.alts, p.b0, 50.0) or abs(
            E.ce_gewinn(gh, gl, p.b0, 50.0) - E.ce_gewinn(kh, kl, p.b0, 50.0)) < 1e-9


def test_kleines_rho_ueberlaeuft_nicht():
    ce = E.ce_gewinn(200.0, -100.0, 0.5, 0.5)
    assert math.isfinite(ce) and -100.0 <= ce <= 200.0 and ce == pytest.approx(-100.0 + 0.5 * math.log(2), abs=1e-6)
    assert E.ce_gewinn(200.0, -100.0, 1.0, 5.0) == pytest.approx(200.0)           # sicher: Sicherheitsäquivalent = Auszahlung
    assert E.ce_gewinn(200.0, -100.0, 0.0, 5.0) == pytest.approx(-100.0)


def test_sicherheitsaequivalent_zwischen_minimum_und_erwartungswert():
    for p in S.ziehe_probleme(12, 100):
        for h, n_ in p.alts:
            ew = p.b0 * h + (1 - p.b0) * n_
            for rho in (10.0, 100.0, 1000.0):
                ce = E.ce_gewinn(h, n_, p.b0, rho)
                assert min(h, n_) - 1e-9 <= ce <= ew + 1e-9


def test_entscheidungswechsel_je_rho_nimmt_ab_und_ties_zaehlen_nicht():
    gezaehlt, erg = E.wechsel_zaehlung(S.ziehe_probleme(20261121, 4000), [20.0, 50.0, 100.0, 300.0])
    w = [erg[r]["wechsel"] for r in (20.0, 50.0, 100.0, 300.0)]
    assert gezaehlt > 1500 and w == sorted(w, reverse=True) and w[-1] > 0
    # eine Entscheidung mit Gleichstand der Erwartungswerte (G und K beide 30): zählt nicht
    gleich = [S.Problem(((60, 0), (30, 30)), 0.5, 0.8, 1.0)]
    assert E.wechsel_zaehlung(gleich, [50.0])[0] == 0


def test_ein_standort_geschlossene_form_gegen_lp_und_baum():
    g = E.ein_standort_geschlossen()
    assert g["x_rp"] == pytest.approx(96.790, abs=2e-3) and g["rp"] == pytest.approx(129.1122, abs=1e-3)
    assert g["eev"] == pytest.approx(129.2444, abs=1e-3) and g["ws"] == 100.0
    assert g["evpi"] == pytest.approx(g["rp"] - g["ws"]) and g["vss"] == pytest.approx(g["eev"] - g["rp"])
    d = E.nachfrage(2000)
    rp_lp, x_lp = E.szenario_lp_ew(d)
    x_baum, rp_baum = E.beste_kapazitaet_ew(d)
    assert rp_lp == pytest.approx(rp_baum, abs=1e-7) and x_lp == pytest.approx(x_baum, abs=1e-4)
    # die 2000 Quantil-Mittelpunkte weichen von der stetigen Verteilung ab (Tail abgeschnitten): gemessen unter 0,05 GE
    assert abs(rp_lp - g["rp"]) < 0.05


def test_nenner_tabelle_ein_standort():
    g = E.ein_standort_geschlossen()
    t = E.nenner_tabelle(g["evpi"], g["vss"], g["rp"], g["ws"], g["eev"])
    assert t["RP"][0] == pytest.approx(22.55, abs=0.01) and t["RP"][1] == pytest.approx(0.102, abs=0.001)
    assert t["WS"][0] == pytest.approx(29.11, abs=0.01) and t["EEV"][0] == pytest.approx(22.53, abs=0.01)


def test_tangenten_unter_der_kurve_und_gleich_an_den_stuetzstellen():
    """Handwerte H7: u(c) = exp(c/ρ), ρ = 100, Tangenten bei 100, 150, 200: konvex, also nie über der Kurve; an den Stützstellen gleich; bei 125 mehr als 0,5 % darunter."""
    rho = 100.0
    stuetz, a, b = E.tangenten(3, rho, 0.0, von=100.0, bis=200.0)
    assert list(stuetz) == [100.0, 150.0, 200.0]
    raster = np.linspace(60.0, 260.0, 1001)
    huelle = np.max(a[:, None] * raster[None, :] + b[:, None], axis=0)
    assert np.all(huelle <= np.exp(raster / rho) + 1e-12)
    assert np.allclose(np.max(a[:, None] * stuetz[None, :] + b[:, None], axis=0), np.exp(stuetz / rho), atol=1e-12)
    assert (np.exp(1.25) - np.max(a * 125.0 + b)) / np.exp(1.25) > 0.005


def test_tangenten_lp_beste_kapazitaet_gegen_fruehere_werte_und_mehrverlust_nicht_negativ():
    """Beste Kapazität x(ρ) bei 2000 Szenarien gleich den Werten der früheren Messung auf 0,002; Mehrverlust des Tangenten-LP nie negativ; mit 64 Tangenten höchstens 0,13 GE."""
    d = E.nachfrage(2000)
    frueher = {500.0: 100.923, 200.0: 108.953, 100.0: 130.856, 50.0: 189.836, 25.0: 242.346}
    for rho, x in frueher.items():
        assert E.beste_kapazitaet_ce(d, rho)[0] == pytest.approx(x, abs=2e-3)
    for rho, m in ((50.0, 8), (100.0, 16), (200.0, 4)):
        x_ex, ce_ex = E.beste_kapazitaet_ce(d, rho)
        x_lp, nv, nz = E.tangenten_lp(d, rho, m, ce_ex)
        assert E.mehrverlust(d, rho, x_lp, ce_ex) > -1e-9 and nv == 1 + 3 * 2000 and nz == 2000 + m * 2000


def test_tangenten_lp_mit_einer_tangente_und_mehrverlust_fuer_rho_50_bei_m64_klein():
    d = E.nachfrage(500)
    x_ex, ce_ex = E.beste_kapazitaet_ce(d, 50.0)
    x1, _, _ = E.tangenten_lp(d, 50.0, 1, ce_ex)
    assert math.isfinite(x1) and x1 >= 0.0
    x64, _, _ = E.tangenten_lp(d, 50.0, 64, ce_ex)
    assert E.mehrverlust(d, 50.0, x64, ce_ex) < 0.1      # bei 500 Szenarien gemessen 0,0095 (rho = 50)

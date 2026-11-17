"""Claims: jede Zahl der README (Tabelle „Befunde“, B1 bis B17) wird hier nachgerechnet, über dieselben Funktionen wie die App.

Tabelle „Zahl → Test“ (Band oder exakt):
  B1 ohne Test, EVPI, Nenner ............ test_b1_ohne_test_evpi_nenner (exakt, Brüche)
  B2 ein Test, EVSI 35, Schwelle 17/30 .. test_b2_ein_test_evsi (exakt)
  B3 zwei Tests, 807/10, Abstand 7/10 ... test_b3_handbeispiel_vier_verfahren (exakt; LP und MIP auf 1e-9 bzw. 1e-5)
  B4 dreistufig 2117/16 ................. test_b4_dreistufig (exakt und LP 1e-9)
  B5 Wachstum N = 0 bis 12 .............. test_b5_wachstum_n0_bis_12 (exakt, Brüche)
  B6 LP gleich Verteilungsrechnung ...... test_b6_lp_gleich_verteilungsrechnung (eingefroren und neu in 300 LPs)
  B7 Stoppen besser, 220 von 1000 ........ test_b7_zufall_hauptmessung_eingefroren (exakt aus der Datei), test_b7_zufall_neu_gerechnet_als_band (± 2)
  B8 vier Zufalls-Seeds, 188 bis 220 ..... test_b8_vier_zufalls_seeds (aus der Datei)
  B9 MIP gleich Baum .................... test_b9_mip_gleich_baum (eingefroren; im Repo neu in 40 Problemen)
  B10 Standardlücke ..................... Messung der README, im Test nur als Band 0 bis 10 (tests/test_frozen.py)
  B11 Zustände gegen Knoten ............. test_b11_zustaende_gegen_knoten (exakt)
  B12 Entscheidungswechsel .............. test_b12_entscheidungswechsel (aus der Datei; neu gerechnet als Band in tests/test_frozen.py)
  B13 Tangenten-LP ...................... test_b13_tangenten_eingefroren (aus der Datei), test_b13_tangenten_neu_gerechnet_m4_m8 (Band)
  B14 Szenarienzahl ..................... test_b14_szenarienzahl (Band)
  B15 Nenner Ein-Standort ............... test_b15_nenner_ein_standort (Band 0,01)
  B16 Nenner Handbeispiel Kosten ........ test_b16_nenner_handbeispiel_kosten (exakt, Brüche)
  B17 Wegweiser, 780 von 1000 ........... test_b17_wegweiser_zaehlung"""

import json
from fractions import Fraction as Fr
from pathlib import Path

import pytest
from scipy import optimize

import evw_constants as C
import evw_evaluation as ev
import evw_exponential as E
import evw_kern as K
import evw_wegweiser as W

ROOT = Path(__file__).resolve().parent.parent
FROZEN = json.loads((ROOT / "evw_frozen.json").read_text(encoding="utf-8"))
H = Fr(1, 2)
Q = Fr(4, 5)


def test_b1_ohne_test_evpi_nenner():
    o = K.ohne_test()
    assert (K.ew(K.A_CAP[0], H), K.ew(K.A_CAP[1], H), o["ws"], o["evpi"]) == (50, 30, 115, 65)
    assert o["evpi_von_rp"] == Fr(13, 10) and o["evpi_von_ws"] == Fr(13, 23) and round(100 * float(o["evpi_von_ws"]), 1) == 56.5


def test_b2_ein_test_evsi():
    assert K.fest_wert(H, K.A_CAP, 0, K.tests_gleich(Q, 1), 1) == 85 and K.evsi_ein_test(H, K.A_CAP, Q) == 35 and Fr(35, 65) == Fr(7, 13)
    assert K.evsi_ein_test(H, K.A_CAP, Fr(17, 30)) == 0 and K.evsi_ein_test(H, K.A_CAP, Fr(17, 30) + Fr(1, 1000)) > 0
    w, nv, neq = K.fest_lp(H, K.A_CAP, 0, K.tests_gleich(Q, 1), 1)
    assert w == pytest.approx(85.0, abs=1e-9) and (nv, neq) == (4, 2)


def test_b3_handbeispiel_vier_verfahren():
    v = ev.analyse_tests(2, 16, 5.0, True)
    assert v.baum_wert == v.dp_wert == Fr(807, 10) and (v.baum_knoten, v.baum_innere, v.dp_zustaende) == (7, 3, 6)
    assert v.mip[1:] == (10, 3, 6) and v.mip[0] == pytest.approx(80.7, abs=1e-5)
    assert v.fest_werte == [Fr(50), Fr(80), Fr(391, 5)] and v.bestes_n == 1 and v.abstand == Fr(7, 10)
    assert K.strategie_kurz(v.strategie) == "Start: Test; nach +: G; nach −: Test; nach −+: G; nach −−: K"
    t = K.tests_gleich(Q, 2)
    assert [K.fest_lp(H, K.A_CAP, Fr(5), t, n)[0] for n in range(3)] == pytest.approx([50.0, 80.0, 78.2], abs=1e-9)


def test_b4_dreistufig():
    d = ev.analyse_dreistufig(100)
    assert d.optimum == Fr(2117, 16) == Fr("132.3125") and d.x1_optimum == 100 and d.knickstellen == 17 and (d.lp_variablen, d.lp_szenarien) == (22, 9)
    assert d.lp_wert == pytest.approx(132.3125, abs=1e-9)


def test_b5_wachstum_n0_bis_12():
    erwartet = {0: Fr(50), 1: Fr(80), 2: Fr(807, 10), 3: Fr("87.8"), 4: Fr("88.024"), 5: Fr("90.296")}
    abstand = {0: 0, 1: 0, 2: Fr(7, 10), 3: Fr("3.4")}
    for N in range(13):
        t = K.tests_gleich(Q, N)
        wert, _ = K.baum(H, K.A_CAP, Fr(5), t)
        w_dp, _, zustaende = K.dp(H, K.A_CAP, Fr(5), t)
        assert wert == w_dp and K.baum_knoten(N)[0] == 2 ** (N + 1) - 1 and zustaende == (N + 1) * (N + 2) // 2, N
        if N in erwartet:
            assert wert == erwartet[N], N
        beste = max(K.fest_wert(H, K.A_CAP, Fr(5), t, n) for n in range(N + 1))
        if N in abstand:
            assert wert - beste == abstand[N], N
        if N == 12:
            assert float(wert) == pytest.approx(91.434448, abs=5e-7) and (K.baum_knoten(N)[0], zustaende) == (8191, 91)
            assert beste == Fr("84.4") and float(wert - beste) == pytest.approx(7.034448, abs=5e-7)
            assert max(range(N + 1), key=lambda n: (K.fest_wert(H, K.A_CAP, Fr(5), t, n), -n)) == 3


def test_b6_lp_gleich_verteilungsrechnung():
    for z in FROZEN["zufall"].values():
        assert z["lp_geprueft"] == 5000 and z["lp_abweichungen"] == 0 and z["lp_max_abweichung"] < 2e-13
    assert sum(z["lp_geprueft"] for z in FROZEN["zufall"].values()) == 20000
    # im Repo neu: tests/test_oracle_szenario_lp.py (300 LPs, 60 Probleme)


def test_b7_zufall_hauptmessung_eingefroren():
    z = FROZEN["zufall"]["20261120"]
    assert (z["nicht_dominiert"], z["test_lohnt"], z["adaptiv_besser"], z["adaptiv_besser_unter_lohnend"]) == (516, 252, 220, 220)
    assert z["test_lohnt"] - z["adaptiv_besser_unter_lohnend"] == 32 and z["dp_abweichungen"] == 0
    assert round(z["luecke_mittel_lohnend"], 2) == 2.17 and round(z["luecke_median_lohnend"], 2) == 1.91 and round(z["luecke_max"], 2) == 9.46 and round(z["luecke_mittel_alle"], 2) == 0.55


def test_b7_zufall_neu_gerechnet_als_band():
    z = ev.zufall_lauf(20261120, 1000, 4, C.ART_ZUFALL, 16, 5.0)
    assert abs(z.nicht_dominiert - 516) <= 2 and abs(z.test_lohnt - 252) <= 2 and abs(z.adaptiv_besser - 220) <= 2 and z.dp_abweichungen == 0
    assert z.besser_unter_lohnend == z.adaptiv_besser
    assert sum(z.luecken_lohnend) / len(z.luecken_lohnend) == pytest.approx(2.172, abs=0.01) and max(z.luecken_lohnend) == pytest.approx(9.456, abs=0.01)


def test_b8_vier_zufalls_seeds():
    z = FROZEN["zufall"]
    assert [z[s]["adaptiv_besser"] for s in ("1", "777", "424242")] == [199, 196, 188] and [z[s]["test_lohnt"] for s in ("1", "777", "424242")] == [232, 231, 212]
    assert min(v["adaptiv_besser"] for v in z.values()) == 188 and max(v["adaptiv_besser"] for v in z.values()) == 220
    assert all(v["lp_abweichungen"] == 0 for v in z.values())


def test_b9_mip_gleich_baum():
    for z in FROZEN["zufall"].values():
        assert z["mip_geprueft"] == 300 and z["mip_abweichungen"] == 0 and z["mip_max_abweichung"] < 1.1e-6
    # im Repo neu: tests/test_oracle_szenario_lp.py (40 Probleme bei N = 4, Handbeispiel N = 2, 3, 5)


def test_b11_zustaende_gegen_knoten():
    tab = ev.zustaende_tabellen()
    zeile = lambda art: [tab[art]["summe"][n] for n in (2, 4, 8, 14)]       # noqa: E731
    assert [K.baum_knoten(n)[0] for n in (2, 4, 8, 14)] == [7, 31, 511, 32767]
    assert zeile(K.TESTARTEN[0]) == [6, 15, 45, 120] and zeile(K.TESTARTEN[1]) == [7, 22, 95, 372] and zeile(K.TESTARTEN[2]) == [7, 31, 511, 32767]
    for art in K.TESTARTEN:
        assert all(tab[art]["summe"][n] == K.zustaende_formel(art, n) for n in range(15))


def test_b12_entscheidungswechsel():
    w = FROZEN["wechsel"]
    assert w["m"] == 20000 and w["seed"] == 20261121 and w["nicht_dominiert"] == 9843
    anteile = {r: round(100 * v["wechsel"] / w["nicht_dominiert"], 1) for r, v in w["rho"].items()}
    assert anteile == {"20": 33.1, "50": 22.9, "100": 13.6, "300": 5.2}
    assert [round(v["verlust_mittel"], 1) for v in w["rho"].values()] == [35.2, 26.6, 16.9, 6.2]
    assert [round(v["verlust_max"], 1) for v in w["rho"].values()] == [189.9, 137.1, 82.2, 24.8]
    g = FROZEN["wechsel_gegenprobe_rho50"]
    assert [round(100 * g[s]["anteil"], 1) for s in ("1", "777", "424242")] == [22.3, 22.0, 22.1]


def test_b13_tangenten_eingefroren():
    z = {(int(r["rho"]), r["m"]): r for r in FROZEN["tangenten"]["zeilen"]}
    assert [round(z[(rho, 4)]["x_exakt"], 3) for rho in (500, 200, 100, 50, 25)] == [100.923, 108.953, 130.856, 189.836, 242.346]
    assert all(z[(rho, 64)]["mehrverlust"] < 0.13 for rho in (500, 200, 100, 50, 25))
    assert round(z[(50, 64)]["mehrverlust"], 3) == 0.047 and round(z[(50, 64)]["ce_exakt"], 1) == 209.8 and round(100 * z[(50, 64)]["mehrverlust"] / z[(50, 64)]["ce_exakt"], 2) == 0.02
    assert round(z[(50, 4)]["mehrverlust"], 1) == 19.1
    assert round(z[(25, 8)]["mehrverlust"], 2) == 1.15 and round(z[(25, 16)]["mehrverlust"], 2) == 2.08 and z[(25, 16)]["mehrverlust"] > z[(25, 8)]["mehrverlust"]
    assert round(z[(200, 8)]["mehrverlust"], 4) == 0.0007 and round(z[(200, 32)]["mehrverlust"], 4) == 0.0116 and z[(200, 32)]["mehrverlust"] > z[(200, 8)]["mehrverlust"]


def test_b13_tangenten_neu_gerechnet_m4_m8():
    d = E.nachfrage(2000)
    z = {(int(r["rho"]), r["m"]): r for r in FROZEN["tangenten"]["zeilen"]}
    for rho, m in ((50, 4), (100, 8), (25, 8)):
        x_ex, ce_ex = E.beste_kapazitaet_ce(d, float(rho))
        x_lp = E.tangenten_lp(d, float(rho), m, ce_ex)[0]
        assert E.mehrverlust(d, float(rho), x_lp, ce_ex) == pytest.approx(z[(rho, m)]["mehrverlust"], abs=0.02 + 0.05 * z[(rho, m)]["mehrverlust"])


def test_b14_szenarienzahl():
    x = {n: E.beste_kapazitaet_ce(E.nachfrage(n), 50.0)[0] for n in (500, 1000, 2000)}
    assert x[500] == pytest.approx(172.4, abs=0.05) and x[1000] == pytest.approx(181.1, abs=0.05) and x[2000] == pytest.approx(189.8, abs=0.05)
    d = E.nachfrage(500)
    x_ex, ce_ex = E.beste_kapazitaet_ce(d, 50.0)
    x_lp = E.tangenten_lp(d, 50.0, 64, ce_ex)[0]
    assert E.mehrverlust(d, 50.0, x_lp, ce_ex) == pytest.approx(0.0095, abs=0.01)


def test_b15_nenner_ein_standort():
    g = E.ein_standort_geschlossen()
    assert (round(g["x_rp"], 3), round(g["rp"], 4), round(g["eev"], 4), g["ws"]) == (96.790, 129.1122, 129.2444, 100.0)
    assert (round(g["evpi"], 4), round(g["vss"], 4)) == (29.1122, 0.1322)
    t = E.nenner_tabelle(g["evpi"], g["vss"], g["rp"], g["ws"], g["eev"])
    assert [round(t[k][0], 2) for k in ("RP", "WS", "EEV")] == [22.55, 29.11, 22.52]
    assert [round(t[k][1], 3) for k in ("RP", "WS", "EEV")] == [0.102, 0.132, 0.102]


def test_b16_nenner_handbeispiel_kosten():
    dd = [50, 100, 200]
    pd = [Fr(1, 4), Fr(1, 2), Fr(1, 4)]
    kosten = lambda x: x + 3 * sum(p * max(d - x, 0) for p, d in zip(pd, dd))     # noqa: E731
    mitt = sum(p * d for p, d in zip(pd, dd))
    rp = min(kosten(Fr(x)) for x in (0, 50, 100, 200))
    ws, eev = mitt, kosten(mitt)
    assert (mitt, rp, ws, eev) == (Fr(225, 2), 175, Fr(225, 2), Fr(1425, 8))
    evpi, vss = rp - ws, eev - rp
    assert (evpi, vss) == (Fr(125, 2), Fr(25, 8)) and evpi / rp == Fr(5, 14) and evpi / ws == Fr(5, 9) and vss / rp == Fr(1, 56)
    assert [round(100 * float(x), 1) for x in (evpi / rp, evpi / ws)] == [35.7, 55.6] and round(100 * float(vss / rp), 2) == 1.79
    res = optimize.linprog([1.0] + [0.75, 1.5, 0.75], A_ub=[[-1, -1, 0, 0], [-1, 0, -1, 0], [-1, 0, 0, -1]], b_ub=[-50, -100, -200], bounds=[(0, None)] * 4, method="highs")
    assert res.fun == pytest.approx(175.0, abs=1e-9) and res.x[0] == pytest.approx(100.0, abs=1e-7)


def test_b17_wegweiser_zaehlung():
    z = FROZEN["zufall"]["20261120"]
    assert 1000 - z["test_lohnt"] == 748 and z["test_lohnt"] - z["adaptiv_besser_unter_lohnend"] == 32 and 748 + 32 == 780 == z["m"] - z["adaptiv_besser"]
    p = ev.wahlregel_pruefung(ev.zufall_lauf(20261120, 1000, 4, C.ART_ZUFALL, 16, 5.0))
    assert abs(p["ueberempfiehlt"] - 780) <= 2 and p["trifft"] == 1000 and abs(p["verfehlt"] - 220) <= 2
    assert len(W.alle_kombinationen()) == 72

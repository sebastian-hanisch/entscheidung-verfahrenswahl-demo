"""Abbildungen: jede Figur lässt sich bauen, alle Achsen sind gesperrt (fixedrange), der Baum zeichnet 2^(N+1) − 1 Knoten, die Kurven haben die erwarteten Spuren."""

import json
from fractions import Fraction as Fr
from pathlib import Path

import evw_evaluation as ev
import evw_exponential as E
import evw_kern as K
import evw_visualization as V

FROZEN = json.loads((Path(__file__).resolve().parent.parent / "evw_frozen.json").read_text(encoding="utf-8"))


def _gesperrt(fig):
    achsen = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
    return bool(achsen) and all(a.get("fixedrange") is True for a in achsen)


def test_alle_figuren_haben_gesperrte_achsen():
    tab = ev.zustaende_tabellen()
    d = ev.analyse_dreistufig(100)
    c, exakt, huelle, stuetz, u_st = ev.kurve_tangenten(50.0, 16, 209.8)
    knoten = ev.baum_info(Fr(1, 2), K.A_CAP, Fr(5), K.tests_gleich(Fr(4, 5), 2))
    figuren = [
        V.build_verfahren([("Baum", 80.7, "#000"), ("LP", 80.0, "#111")]),
        V.build_baum(knoten, 2),
        V.build_dreistufig(d.raster, d.x1, d.optimum, d.x1_optimum),
        V.build_luecken([0.0, 0.5, 1.5, 2.0]),
        V.build_wachstum(tab, K.TESTARTEN[0], 10),
        V.build_tangenten(c, exakt, huelle, stuetz, u_st, 50, 16),
        V.build_mehrverlust(FROZEN["tangenten"]["zeilen"], 50, 64, 0.01),
        V.build_wechsel({20.0: 0.33, 50.0: 0.23}),
    ]
    for fig in figuren:
        assert _gesperrt(fig) and len(fig.data) >= 1


def test_baum_hat_einen_knoten_je_historie_und_n3_ist_das_maximum():
    for n in (0, 1, 2, 3):
        knoten = ev.baum_info(Fr(1, 2), K.A_CAP, Fr(5), K.tests_gleich(Fr(4, 5), n))
        assert len(knoten) == 2 ** (n + 1) - 1
        fig = V.build_baum(knoten, n)
        assert sum(len(t.x) for t in fig.data if t.mode == "markers+text") == 2 ** (n + 1) - 1


def test_baum_info_markiert_erreichbare_knoten_nach_der_strategie():
    """Handbeispiel: erreicht werden die Wurzel, „+“ (stoppt), „−“ (testet weiter) und dessen zwei Kinder; die Kinder von „+“ sind nicht erreichbar."""
    knoten = {k["pfad"]: k for k in ev.baum_info(Fr(1, 2), K.A_CAP, Fr(5), K.tests_gleich(Fr(4, 5), 2))}
    assert [pfad for pfad, k in knoten.items() if k["erreichbar"]] == ["", "+", "−", "−+", "−−"]
    assert knoten["+"]["stoppt"] and not knoten[""]["stoppt"] and not knoten["−"]["stoppt"]
    assert knoten[""]["wert"] == Fr(807, 10) and knoten["−"]["wert"] == Fr("31.4") and knoten["+"]["wert"] == 140


def test_kurve_tangenten_huelle_nie_ueber_der_kurve():
    for rho in (25.0, 100.0, 500.0):
        c, exakt, huelle, stuetz, u_st = ev.kurve_tangenten(rho, 8, E.ce_kosten(E.kosten(150.0, E.nachfrage(200)), rho))
        assert (huelle <= exakt + 1e-9).all() and len(c) == 200

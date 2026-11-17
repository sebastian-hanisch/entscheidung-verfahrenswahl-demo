"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt fertige Zeilen, nicht Zufalls-Seeds; so prüft der Test die Grenzen und jede genannte Zahl direkt."""

from fractions import Fraction as Fr

import evw_evaluation as ev
import evw_wegweiser as W

FROZEN_MINI = {"zufall": {"a": {"adaptiv_besser": 188}, "b": {"adaptiv_besser": 220}}, "wechsel": {"seed": 20261121, "m": 20000, "nicht_dominiert": 10000,
                                                                                              "rho": {"20": {"wechsel": 3000}, "50": {"wechsel": 2000}}}}


def _lauf(m=100, lohnt=40, besser=30, besser_l=30, dp_abw=0, luecken_l=(1.0, 2.0, 3.0)):
    return ev.ZufallsLauf(seed=7, m=m, n_tests=4, art="x", nicht_dominiert=60, test_lohnt=lohnt, adaptiv_besser=besser, besser_unter_lohnend=besser_l, dp_abweichungen=dp_abw,
                          luecken=[0.0] * (m - len(luecken_l)) + list(luecken_l), luecken_lohnend=list(luecken_l), mittel_baum_lohnend=90.0, mittel_fest_lohnend=88.0, index_max=0)


def test_formatierung_mit_komma_und_bruch():
    assert ev.de(3.14159, 2) == "3,14" and ev.de_kurz(80.7) == "80,7" and ev.de_kurz(50) == "50" and ev.de_kurz(-0.00001) == "0"
    assert ev.wert_text(Fr(807, 10)) == "80,7 (807/10)" and ev.wert_text(Fr(50)) == "50" and ev.pct(Fr(13, 23)) == "56,5 %" and ev.pct(Fr(13, 10), 0) == "130 %"
    assert ev.tausender(32767) == "32 767" and ev.tausender(120) == "120"


def test_verdict_verfahren_stoppen_mit_abstand_nennt_wert_abstand_evpi_und_beide_nenner():
    kind, text = ev.verdict_verfahren(ev.analyse_tests(2, 16, 5.0, True))
    assert kind == "success"
    assert "**80,7 (807/10) GE**" in text and "höchstens 80 GE (n = 1)" in text and "**Abstand 0,7 GE**" in text
    assert "EVPI 65 GE = 56,5 % von WS (RP = 50 GE, WS = 115 GE" in text and "130 %" not in text and "gemischt-ganzzahlige Modell 80,7" in text


def test_verdict_verfahren_grenze_abstand_1e_minus_9():
    v = ev.analyse_tests(1, 16, 5.0, True)                  # N = 1: Baum = bestes festes n = 80
    assert v.abstand == 0 and ev.verdict_verfahren(v)[0] == "info"
    v.abstand = Fr(1, 10 ** 9)
    assert ev.verdict_verfahren(v)[0] == "info"             # genau 1e-9 zählt noch als gleich
    v.abstand = Fr(1, 10 ** 9) + Fr(1, 10 ** 12)
    assert ev.verdict_verfahren(v)[0] == "success"


def test_verdict_verfahren_feste_testzahl():
    kind, text = ev.verdict_verfahren(ev.analyse_tests(2, 16, 5.0, False))
    assert kind == "success" and "**80 GE** bei n = 1 Tests" in text and "Abstand höchstens 1e-9" in text


def test_verdict_verfahren_ohne_mip_ab_n9_sagt_nicht_gerechnet():
    v = ev.analyse_tests(9, 16, 5.0, True)
    assert v.mip is None and "nicht gerechnet" in [z for z in ev.zeilen_verfahren(v) if z["Verfahren"].startswith("gemischt")][0]["Wert (GE)"]
    assert "gemischt-ganzzahlige Modell" not in ev.verdict_verfahren(v)[1]


def test_verdict_zufall_nennt_zaehlung_mittel_groesste_und_vorgerechnete_spanne():
    kind, text = ev.verdict_zufall(_lauf(), FROZEN_MINI)
    assert kind == "success"
    assert "Von 100 Zufallsproblemen (Zufalls-Seed 7, bis zu 4 gleiche Tests nacheinander) ist Stoppen nach Beobachtung in **30** (30,0 %)" in text
    assert "in 40 lohnt sich überhaupt ein Test" in text and "im Mittel 2,00 GE (größte 3,00 GE)" in text and "in 10 davon reicht eine feste Zahl" in text
    assert "188 bis 220 von 1000" in text and "in 0 Problemen ab" in text


def test_verdict_zufall_ohne_lohnenden_test_ist_eine_warnung():
    kind, text = ev.verdict_zufall(_lauf(lohnt=0, besser=0, besser_l=0, luecken_l=()), FROZEN_MINI)
    assert kind == "warning" and "In keinem der 100 Zufallsprobleme" in text


def test_verdict_zustaende_nennt_knoten_und_drei_arten_und_schwelle():
    tab = ev.zustaende_tabellen()
    kind, text = ev.verdict_zustaende(14, ev.K.TESTARTEN[0], tab)
    assert kind == "info" and "N = 14: 32 767 Baum-Knoten" in text and "gleichen Tests 120" in text and "zwei Sorten abwechselnd 372" in text and "Primzahl-Quoten (multiplikativ unabhängige Quotenverhältnisse) 32 767" in text
    assert "120 Zustände = 0,4 % der Knoten" in text
    kind, text = ev.verdict_zustaende(14, ev.K.TESTARTEN[2], tab)
    assert kind == "warning" and "32 767 Zustände = 100,0 % der Knoten" in text


def test_verdict_wegweiser_nur_fuer_stoppen_definiert():
    p = {"m": 1000, "verfehlt": 220, "trifft": 1000, "ueberempfiehlt": 780, "kein_test": 748, "fest_reicht": 32, "lohnt": 252}
    stoppen = W.wegweiser(W.INFO[2], W.NUTZEN[0], W.ZUSTAND[0], W.QUELLE[0], W.EU[0])
    kind, text = ev.verdict_wegweiser(stoppen, p)
    assert kind == "success" and "an 1000 Zufallsproblemen" in text and "**220**" in text and "in 1000" in text and "in den 780 anderen (748 ohne lohnenden Test, 32 mit ausreichender" in text
    kind, text = ev.verdict_wegweiser(W.wegweiser(W.INFO[0], W.NUTZEN[0], W.ZUSTAND[0], W.QUELLE[0], W.EU[0]), p)
    assert kind == "info" and "nicht gerechnet" in text


def test_wahlregel_pruefung_zaehlt_aus_dem_lauf():
    p = ev.wahlregel_pruefung(_lauf())
    assert p == {"m": 100, "verfehlt": 30, "trifft": 100, "ueberempfiehlt": 70, "kein_test": 60, "fest_reicht": 10, "lohnt": 40}


def test_kernfragen_nennen_abstand_zaehlung_und_zustaende():
    tab = ev.zustaende_tabellen()
    a = ev.kernfragen(ev.analyse_tests(2, 16, 5.0, True), _lauf(), 14, ev.K.TESTARTEN[0], tab)
    assert len(a) == 3 and all(k == "info" for k, _ in a)
    assert "80,7 (807/10) GE; das beste Szenario-LP mit fester Testzahl bleibt 0,7 GE darunter" in a[0][1]
    assert "In 30 von 100 Zufallsproblemen (Zufalls-Seed 7)" in a[1][1] and "120 Zustände gegen 32 767 Baum-Knoten" in a[2][1] and "gleiche Tests" in a[2][1]


def test_verdict_dreistufig_zweig_ohne_optimum():
    d = ev.analyse_dreistufig(80)
    kind, text = ev.verdict_dreistufig(d)
    assert kind == "info" and "x₁ = 80 kostet der Baum im Erwartungswert 133,7812 (4281/32) GE" in text and "alle 17 Knickstellen" in text

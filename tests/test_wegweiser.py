"""Wegweiser: Wahrheitstabelle über alle 72 Kombinationen der fünf Merkmale; jede Ausgabe mit Voraussetzung und existierender Belegzeile; Formfrage zuerst
(die Größe kippt die Empfehlung nie); „nicht gemessen“ und „keines der drei“ als eigene Ausgaben."""

import inspect
import json
import re
from pathlib import Path

import pytest

import evw_wegweiser as W

ROOT = Path(__file__).resolve().parent.parent
FROZEN = json.loads((ROOT / "evw_frozen.json").read_text(encoding="utf-8"))


def erwartete_art(info, nutzen, zustand, quelle, eu):
    """Unabhängige Fassung der Rangfolge: Allais vor Stichprobe vor Nutzen; die Größe kommt nicht vor."""
    if eu == W.EU[1]:
        return W.ART_KEINES
    if quelle == W.QUELLE[1]:
        return W.ART_VERFAHREN
    if nutzen == W.NUTZEN[2]:
        return W.ART_NICHT_GEMESSEN
    return W.ART_VERFAHREN


def test_72_kombinationen_jede_mit_voraussetzung_beleg_und_gegenfall():
    alle = W.alle_kombinationen()
    assert len(alle) == 3 * 3 * 2 * 2 * 2 == 72 and len({k for k, _ in alle}) == 72
    for kombi, e in alle:
        assert e.art == erwartete_art(*kombi), kombi
        assert e.voraussetzung.strip() and e.gegenfall.strip() and e.verfahren.strip()
        assert e.beleg_id in W.BELEG_IDS
        zeilen = W.belegzeilen(e, FROZEN)
        assert zeilen and all(len(z) > 40 for z in zeilen)


def test_jede_belegzeile_nennt_nur_vorhandene_testdateien():
    for kombi, e in W.alle_kombinationen():
        for zeile in W.belegzeilen(e, FROZEN):
            for pfad in re.findall(r"`(tests/[a-z_]+\.py)`", zeile):
                assert (ROOT / pfad).exists(), pfad
    for beleg_id in W.BELEG_IDS:
        assert W.beleg_text(beleg_id, FROZEN)


def test_wegweiser_hat_keine_groesse_als_eingang():
    """Formfrage zuerst: die fünf Merkmale sind die einzigen Parameter; kein N, keine Knotenzahl."""
    namen = list(inspect.signature(W.wegweiser).parameters)
    assert namen == ["info", "nutzen", "zustand", "quelle", "eu"]


def test_gleiche_form_gleiche_empfehlung_andere_informationsstruktur_andere_empfehlung():
    basis = dict(nutzen=W.NUTZEN[0], zustand=W.ZUSTAND[0], quelle=W.QUELLE[0], eu=W.EU[0])
    e = [W.wegweiser(info, **basis) for info in W.INFO]
    assert len({x.verfahren for x in e}) == 3 and len({x.beleg_id for x in e}) == 3
    assert W.wegweiser(W.INFO[2], **basis) == W.wegweiser(W.INFO[2], **basis)


def test_stoppen_zustandsmerkmal_aendert_verfahren_und_belegzeile():
    kw = dict(nutzen=W.NUTZEN[0], quelle=W.QUELLE[0], eu=W.EU[0])
    ja, nein = W.wegweiser(W.INFO[2], zustand=W.ZUSTAND[0], **kw), W.wegweiser(W.INFO[2], zustand=W.ZUSTAND[1], **kw)
    assert "spart nichts" in nein.verfahren and "spart nichts" not in ja.verfahren
    assert "120 Zustände" in " ".join(W.belegzeilen(ja, FROZEN)) and "32 767 Zustände" in " ".join(W.belegzeilen(nein, FROZEN))
    assert ja.beleg_id == nein.beleg_id == "B_STOPPEN"


def test_zustandsmerkmal_ohne_stoppen_aendert_nichts():
    kw = dict(nutzen=W.NUTZEN[0], quelle=W.QUELLE[0], eu=W.EU[0])
    for info in W.INFO[:2]:
        assert W.wegweiser(info, zustand=W.ZUSTAND[0], **kw) == W.wegweiser(info, zustand=W.ZUSTAND[1], **kw)


def test_nicht_gemessen_und_keines_der_drei_sind_eigene_ausgaben():
    nicht = W.wegweiser(W.INFO[0], W.NUTZEN[2], W.ZUSTAND[0], W.QUELLE[0], W.EU[0])
    assert nicht.art == W.ART_NICHT_GEMESSEN and nicht.beleg_id == "B_ANDERER" and "Nicht gemessen" in W.beleg_text("B_ANDERER", FROZEN)
    keines = W.wegweiser(W.INFO[2], W.NUTZEN[0], W.ZUSTAND[0], W.QUELLE[0], W.EU[1])
    assert keines.art == W.ART_KEINES and keines.beleg_id == "B_ALLAIS" and "nicht gemessen" in W.beleg_text("B_ALLAIS", FROZEN)
    assert all(e.art == W.ART_KEINES for k, e in W.alle_kombinationen() if k[4] == W.EU[1])


def test_belegzeilen_zahlen_stammen_aus_den_eingefrorenen_messungen():
    z = FROZEN["zufall"]["20261120"]
    text = W.beleg_text("B_STOPPEN", FROZEN)
    assert f"{z['adaptiv_besser']} von {z['m']}" in text
    besser = [v["adaptiv_besser"] for v in FROZEN["zufall"].values()]
    assert f"{min(besser)} bis {max(besser)}" in text
    assert str(z["lp_geprueft"]) in W.beleg_text("B_FEST", FROZEN)
    w = FROZEN["wechsel"]["rho"]["50"]
    assert f"{100 * w['wechsel'] / FROZEN['wechsel']['nicht_dominiert']:.1f}".replace(".", ",") + " %" in W.beleg_text("B_EXP", FROZEN)


@pytest.mark.parametrize("name", ["Entscheidungsbaum"])
def test_der_wegweiser_heisst_nicht_entscheidungsbaum(name):
    """„Entscheidungsbaum“ steht in dieser Demo nur als Rechenverfahren „Entscheidungsbaum (Ereignisknoten)“ (CART ist etwas anderes)."""
    quelle = (ROOT / "evw_wegweiser.py").read_text(encoding="utf-8")
    for treffer in re.finditer(name, quelle):
        umfeld = quelle[max(0, treffer.start() - 60): treffer.end() + 60]
        assert "Ereignisknoten" in umfeld or "CART" in umfeld or "Rechenverfahren" in umfeld

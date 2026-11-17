"""Eingefrorene Messreihen (`evw_frozen.json`): Aufbau, innere Stimmigkeit und `tools/evw_make_frozen.py --check` (schnelle Teile neu gerechnet, Zählungen als Band).

Die Datei ist vorgerechnet, weil die Live-Rechnung der App für 20 000 LPs zu lang wäre. Jede Zahl der README aus dieser Datei steht in tests/test_claims.py."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FROZEN = json.loads((ROOT / "evw_frozen.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def werkzeug():
    spec = importlib.util.spec_from_file_location("evw_make_frozen", ROOT / "tools" / "evw_make_frozen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_aufbau_vier_zufalls_seeds_wechsel_und_25_tangenten_zeilen():
    assert set(FROZEN["zufall"]) == {"20261120", "1", "777", "424242"}
    assert all(z["m"] == 1000 and z["N"] == 4 for z in FROZEN["zufall"].values())
    assert FROZEN["wechsel"]["m"] == 20000 and set(FROZEN["wechsel"]["rho"]) == {"20", "50", "100", "300"}
    assert len(FROZEN["tangenten"]["zeilen"]) == 25 and FROZEN["tangenten"]["n"] == 2000
    assert {(r["rho"], r["m"]) for r in FROZEN["tangenten"]["zeilen"]} == {(float(a), b) for a in (500, 200, 100, 50, 25) for b in (4, 8, 16, 32, 64)}


def test_innere_stimmigkeit_der_zaehlungen():
    for z in FROZEN["zufall"].values():
        assert z["adaptiv_besser"] <= z["test_lohnt"] <= z["nicht_dominiert"] <= z["m"]
        assert z["adaptiv_besser_unter_lohnend"] == z["adaptiv_besser"]
        assert z["dp_abweichungen"] == 0 and z["lp_abweichungen"] == 0 and z["mip_abweichungen"] == 0
        assert z["lp_geprueft"] == 5 * z["m"] and z["mip_geprueft"] == 300
        assert z["lp_max_abweichung"] < 1e-7 and z["mip_max_abweichung"] < 1e-5 and z["luecke_min"] > -1e-9
    w = FROZEN["wechsel"]["rho"]
    anteile = [w[r]["wechsel"] / FROZEN["wechsel"]["nicht_dominiert"] for r in ("20", "50", "100", "300")]
    assert anteile == sorted(anteile, reverse=True)


def test_standardluecke_des_mip_weicht_einzeln_nach_unten_ab():
    """Mit der HiGHS-Standardlücke (1e-4) liegen einzelne Werte unter dem Baumwert; die Abweichung ist klein (Toleranz der Software, kein Befund über das Modell). Zählung als Band."""
    s = FROZEN["mip_standardluecke"]
    assert s["geprueft"] == 300 and 0 <= s["abweichungen"] <= 10 and s["groesste_abweichung"] < 0.01


def test_tangenten_zeilen_stimmig():
    for r in FROZEN["tangenten"]["zeilen"]:
        assert r["mehrverlust"] > -1e-9 and r["x_exakt"] > 0 and r["ce_exakt"] > 0


def test_check_der_schnellen_teile_findet_keine_abweichung(werkzeug):
    assert werkzeug.pruefe(werkzeug.laden()) == []

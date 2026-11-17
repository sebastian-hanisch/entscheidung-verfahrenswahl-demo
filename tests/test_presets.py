"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, je Preset die erwartete Wirkung (kein Preset ohne Wirkung meldet eine Wirkung)."""

import json
from pathlib import Path

import pytest

import evw_constants as C
import evw_evaluation as ev
import evw_exponential as E
import evw_presets as P

FROZEN = json.loads((Path(__file__).resolve().parent.parent / "evw_frozen.json").read_text(encoding="utf-8"))


def _wert(preset, key):
    return C.PRESETS[preset].get(key, P.SETTING_SPECS[key].default)


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.SETTING_SPECS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_or_options():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_values_are_inside_the_bounds_or_options():
    for key, spec in P.SETTING_SPECS.items():
        assert (spec.default in spec.options) if spec.options is not None else (spec.lo <= spec.default <= spec.hi), key


def test_first_preset_is_the_default_settings_and_all_presets_differ():
    erstes = next(iter(C.PRESETS))
    assert C.PRESETS[erstes] == {}
    vollstaendig = [tuple(_wert(n, k) for k in P.SETTING_SPECS) for n in C.PRESETS]
    assert len(set(vollstaendig)) == len(C.PRESETS)


def test_anderer_wert_liefert_gueltige_werte_fuer_jeden_regler():
    for key, spec in P.SETTING_SPECS.items():
        neu = P.anderer_wert(key, spec.default)
        assert neu != spec.default and ((neu in spec.options) if spec.options is not None else (spec.lo <= neu <= spec.hi)), key


def test_url_parameter_sind_eindeutig():
    namen = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(namen) == len(set(namen))


def test_preset_handbeispiel_80_7_und_abstand_0_7():
    v = ev.analyse_tests(_wert("Handbeispiel (zwei Tests, 80,7)", "e1_n"), _wert("Handbeispiel (zwei Tests, 80,7)", "e1_q20"), _wert("Handbeispiel (zwei Tests, 80,7)", "e1_c"), True)
    kind, text = ev.verdict_verfahren(v)
    assert kind == "success" and "80,7 (807/10) GE" in text and "Abstand 0,7 GE" in text and "130 %" not in text and "56,5 % von WS" in text


def test_preset_ein_test_evsi_35_alle_verfahren_gleich_auch_bei_fester_testzahl():
    name = "Ein Test (EVSI 35)"
    for stoppen in (True, False):
        v = ev.analyse_tests(_wert(name, "e1_n"), _wert(name, "e1_q20"), _wert(name, "e1_c"), stoppen)
        assert v.evsi_ein_test == 35 and v.baum_wert == 85 and v.bester_fest == 85 and v.abstand == 0
        kind, text = ev.verdict_verfahren(v)
        assert (kind == "info" and "Hier reicht eine feste Testzahl" in text) if stoppen else (kind == "success" and "denselben Wert" in text)


def test_preset_dreistufig_x1_100_ist_das_optimum():
    d = ev.analyse_dreistufig(_wert("Dreistufig (Kapazität, Nachbesserung)", "e1_x1"))
    kind, text = ev.verdict_dreistufig(d)
    assert kind == "success" and "2117/16" in text and "Abstand höchstens 1e-9" in text
    assert ev.verdict_dreistufig(ev.analyse_dreistufig(C.E1_X1_DEFAULT))[0] == "info"        # bei x₁ = 80 ist das Optimum nicht erreicht


def test_preset_zehn_tests_knoten_gegen_zustaende():
    n = _wert("Zehn Tests (Knoten gegen Zustände)", "e3_n")
    kind, text = ev.verdict_zustaende(n, _wert("Zehn Tests (Knoten gegen Zustände)", "e3_art"), ev.zustaende_tabellen())
    assert n == 10 and "2 047 Baum-Knoten" in text and "gleichen Tests 66" in text and "zwei Sorten abwechselnd 161" in text and "Primzahl-Quoten (multiplikativ unabhängige Quotenverhältnisse) 2 047" in text
    assert kind == "info"


def test_preset_exponentialnutzen_rho_50_m64_vorgerechnet():
    name = "Exponentialnutzen ρ = 50"
    rho, m, szen = _wert(name, "e4_rho"), _wert(name, "e4_m"), _wert(name, "e4_szen")
    t = ev.tangenten_ergebnis(rho, m, szen, FROZEN)
    assert (rho, m, szen) == (50, 64, 2000) and t["live"] is False and t["mehrverlust"] == pytest.approx(0.0468, abs=0.003)
    w = ev.wechsel_live(20261121, 20000, 50)
    kind, text = ev.verdict_exponential(rho, m, t, w, FROZEN)
    assert "22,9 %" in text and "Mehrverlust **0,047 GE**" in text and "vorgerechnet" in text


def test_preset_zufallsprobleme_1000_adaptiv_besser_als_bestes_festes_n_im_band():
    name = "Zufallsprobleme (1000)"
    z = ev.zufall_lauf(_wert(name, "seed_input"), _wert(name, "e2_m"), _wert(name, "e2_n"), _wert(name, "e2_art"), 16, 5.0)
    assert z.m == 1000 and 218 <= z.adaptiv_besser <= 222 and 250 <= z.test_lohnt <= 254
    kind, text = ev.verdict_zufall(z, FROZEN)
    assert kind == "success" and "Von 1000 Zufallsproblemen" in text and "188 bis 220 von 1000" in text


def test_tangenten_live_bei_500_szenarien_gleich_der_lp_rechnung():
    t = ev.tangenten_ergebnis(100, 8, 500, FROZEN)
    assert t["live"] is True and t["szenarien"] == 500 and t["mehrverlust"] > -1e-9 and t["x_lp"] > 0
    d = E.nachfrage(500)
    x_ex, ce_ex = E.beste_kapazitaet_ce(d, 100.0)
    assert t["x_exakt"] == pytest.approx(x_ex) and t["mehrverlust"] == pytest.approx(E.mehrverlust(d, 100.0, t["x_lp"], ce_ex))

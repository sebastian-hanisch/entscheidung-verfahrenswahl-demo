"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Jeder Regler steht in SETTING_SPECS (Adresszeile, Hervorhebung des aktiven Presets). Ein Preset setzt ALLE Regler: die nicht genannten auf ihren Standardwert,
damit der Zustand nach dem Anklicken genau dem Preset entspricht."""

import random
from dataclasses import dataclass

import streamlit as st

import evw_constants as C
import evw_wegweiser as W


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None     # erlaubte Werte (Auswahlregler); sonst gelten lo und hi


SETTING_SPECS = {
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "e1_n": SettingSpec("n1", int, C.E1_N_DEFAULT, C.E1_N_MIN, C.E1_N_MAX),
    "e1_q20": SettingSpec("q20", int, C.E1_Q20_DEFAULT, C.E1_Q20_MIN, C.E1_Q20_MAX),
    "e1_c": SettingSpec("c", float, C.E1_C_DEFAULT, C.E1_C_MIN, C.E1_C_MAX),
    "e1_stoppen": SettingSpec("stop", str, C.STOPPEN_JA, options=C.STOPPEN_OPTIONEN),
    "e1_x1": SettingSpec("x1", int, C.E1_X1_DEFAULT, C.E1_X1_MIN, C.E1_X1_MAX),
    "e2_n": SettingSpec("n2", int, C.E2_N_DEFAULT, C.E2_N_MIN, C.E2_N_MAX),
    "e2_m": SettingSpec("m2", int, C.E2_M_DEFAULT, options=C.E2_M_OPTIONEN),
    "e2_art": SettingSpec("art2", str, C.ART_ZUFALL, options=C.E2_ART_OPTIONEN),
    "e3_n": SettingSpec("n3", int, C.E3_N_DEFAULT, C.E3_N_MIN, C.E3_N_MAX),
    "e3_art": SettingSpec("art3", str, C.E3_ART_OPTIONEN[0], options=C.E3_ART_OPTIONEN),
    "e4_rho": SettingSpec("rho", int, C.E4_RHO_DEFAULT, options=C.E4_RHO_OPTIONEN),
    "e4_m": SettingSpec("m4", int, C.E4_M_DEFAULT, options=C.E4_M_OPTIONEN),
    "e4_szen": SettingSpec("szen", int, C.E4_SZEN_DEFAULT, options=C.E4_SZEN_OPTIONEN),
    "e4_probleme": SettingSpec("prob", int, C.E4_PROBLEME_DEFAULT, options=C.E4_PROBLEME_OPTIONEN),
    "e5_info": SettingSpec("info", str, W.INFO[2], options=W.INFO),
    "e5_nutzen": SettingSpec("nutzen", str, W.NUTZEN[0], options=W.NUTZEN),
    "e5_zustand": SettingSpec("zustand", str, W.ZUSTAND[0], options=W.ZUSTAND),
    "e5_quelle": SettingSpec("quelle", str, W.QUELLE[0], options=W.QUELLE),
    "e5_eu": SettingSpec("eu", str, W.EU[0], options=W.EU),
}


def anderer_wert(state_key, aktuell):
    """Ein gültiger Wert ungleich `aktuell` (für Tests und für Aufräumarbeiten)."""
    spec = SETTING_SPECS[state_key]
    if spec.options is not None:
        return next(o for o in spec.options if o != aktuell)
    return spec.hi if aktuell != spec.hi else spec.lo


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _begrenze(spec, value):
    if spec.options is not None:
        return value if value in spec.options else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt (ungültige Auswahlwerte werden ignoriert)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = _begrenze(spec, spec.caster(qp[spec.url_param]))
            except (ValueError, TypeError):
                continue
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Alle Regler auf die Werte des Presets setzen, die nicht genannten auf ihren Standardwert."""
    preset = C.PRESETS[name]
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = preset.get(state_key, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)

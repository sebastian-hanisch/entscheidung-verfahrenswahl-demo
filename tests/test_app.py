"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


SONDERFAELLE = [
    {"e2_art": "Testgüte und Kosten wie in Experiment 1", "e1_q20": 10},            # q = 1/2: kein Test lohnt, nichts zu zeigen (leere Verdikte)
    {"e1_stoppen": "feste Testzahl (vorab festgelegt)", "e1_n": 10},
    {"e1_n": 0},
    {"e1_n": 10, "e1_q20": 20, "e1_c": 0.0},                                          # vollkommener Test, N über der MIP-Grenze
    {"e3_art": "paarweise verschieden (Primzahl-Quoten)", "e3_n": 14},
    {"e3_art": "zwei Sorten abwechselnd", "e3_n": 5},
    {"e4_szen": 2000, "e4_m": 4, "e4_rho": 25},
    {"e4_szen": 250, "e4_m": 64, "e4_rho": 500},
    {"e5_eu": "nein: Allais-Struktur der Wahlen"},
    {"e5_nutzen": "anderer Nutzen"},
    {"e5_quelle": "Stichprobe aus unbekannter Verteilung"},
    {"e5_info": "keine Beobachtung vor der Entscheidung"},
    {"e2_n": 1, "e2_m": 1000},
]


def test_sonderfaelle_laufen_ohne_ausnahme():
    for zustand in SONDERFAELLE:
        at = AppTest.from_file(APP_PATH)
        for key, value in zustand.items():
            at.session_state[key] = value
        at.run(timeout=TIMEOUT)
        assert not at.exception, (zustand, [str(e) for e in at.exception])

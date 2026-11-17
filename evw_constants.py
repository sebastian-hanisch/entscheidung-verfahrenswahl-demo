"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

from evw_kern import TESTARTEN

# --- Zufall und Zufallsprobleme ---------------------------------------------------------------------------------------------------------------
SEED_MAX, DEFAULT_SEED = 999_999_999, 20261120

# --- Experiment 1: dieselbe Entscheidung in vier Verfahren ------------------------------------------------------------------------------------------
E1_N_MIN, E1_N_MAX, E1_N_DEFAULT = 0, 10, 2
E1_Q20_MIN, E1_Q20_MAX, E1_Q20_DEFAULT = 10, 20, 16          # q = Zähler/20: 1/2 bis 1 im Raster 1/20, Voreinstellung 4/5
E1_C_MIN, E1_C_MAX, E1_C_DEFAULT, E1_C_STEP = 0.0, 10.0, 5.0, 0.5
STOPPEN_JA = "Stoppen nach jedem Ergebnis erlaubt"
STOPPEN_NEIN = "feste Testzahl (vorab festgelegt)"
STOPPEN_OPTIONEN = (STOPPEN_JA, STOPPEN_NEIN)
MIP_N_MAX = 8                                                # darüber wird das gemischt-ganzzahlige Modell nicht gerechnet (Laufzeit über 2 s bei N = 9 und 10)
BAUM_BILD_N_MAX = 3                                          # der Baum wird bis N = 3 als Diagramm gezeigt
E1_X1_MIN, E1_X1_MAX, E1_X1_DEFAULT, E1_X1_STEP = 0, 200, 80, 5

# --- Experiment 2: Zufallsprobleme ------------------------------------------------------------------------------------------------------------
E2_N_MIN, E2_N_MAX, E2_N_DEFAULT = 1, 4, 4
E2_M_OPTIONEN, E2_M_DEFAULT = (200, 500, 1000), 200
ART_ZUFALL = "Testgüte und Kosten zufällig je Problem"
ART_FEST = "Testgüte und Kosten wie in Experiment 1"
E2_ART_OPTIONEN = (ART_ZUFALL, ART_FEST)
E2_BAND_REFERENZ = ("20261120", "1", "777", "424242")        # die vier Zufalls-Seeds der vorgerechneten Messreihe

# --- Experiment 3: Knoten gegen Zustände ------------------------------------------------------------------------------------------------------
E3_N_MIN, E3_N_MAX, E3_N_DEFAULT = 1, 14, 14
E3_ART_OPTIONEN = TESTARTEN
E3_TABELLE_N_MAX = 5                                         # die Tabelle der Glaubenszustände je Tiefe (Brüche) bis N = 5

# --- Experiment 4: Exponentialnutzen im LP ----------------------------------------------------------------------------------------------------
E4_RHO_OPTIONEN, E4_RHO_DEFAULT = (25, 50, 100, 200, 500), 100
E4_M_OPTIONEN, E4_M_DEFAULT = (4, 8, 16, 32, 64), 32
E4_SZEN_OPTIONEN, E4_SZEN_DEFAULT = (250, 500, 2000), 500    # 2000 ist vorgerechnet (evw_frozen.json), 250 und 500 werden live gerechnet
E4_SZEN_VORGERECHNET = 2000
E4_PROBLEME_OPTIONEN, E4_PROBLEME_DEFAULT = (1000, 5000, 20000), 5000
E4_WECHSEL_RHOS = (20.0, 50.0, 100.0, 300.0)
E4_SEED_VERSATZ = 1                                          # die Probleme des Entscheidungswechsels kommen aus Zufalls-Seed + 1 (voreingestellt 20261121)

# --- Experiment 5: Wegweiser (Optionen stehen in evw_wegweiser.py) ----------------------------------------------------------------------------

# --- Darstellung ------------------------------------------------------------------------------------------------------------------------------
COLOR_BAUM = "#2F6B65"
COLOR_LP = "#4c78a8"
COLOR_ZUSTAND = "#e08a1e"
COLOR_GRAU = "rgba(150,150,150,0.55)"
COLOR_ROT = "#c0392b"
COLOR_ARTEN = {TESTARTEN[0]: "#2F6B65", TESTARTEN[1]: "#e08a1e", TESTARTEN[2]: "#8e44ad"}

PRESETS = {
    "Handbeispiel (zwei Tests, 80,7)": {},
    "Ein Test (EVSI 35)": {"e1_n": 1, "e1_c": 0.0},
    "Dreistufig (Kapazität, Nachbesserung)": {"e1_x1": 100},
    "Zehn Tests (Knoten gegen Zustände)": {"e3_n": 10},
    "Exponentialnutzen ρ = 50": {"e4_rho": 50, "e4_m": 64, "e4_szen": 2000},
    "Zufallsprobleme (1000)": {"e2_m": 1000},
}
PRESET_HELP = {
    "Handbeispiel (zwei Tests, 80,7)": "Voreinstellung: höchstens zwei Tests nacheinander, q = 4/5, 5 GE je Test. Baum, Zustandsrechnung und gemischt-ganzzahliges Modell geben 80,7; "
                                       "bei fester Testzahl gibt das Szenario-LP höchstens 80.",
    "Ein Test (EVSI 35)": "Ein einziger Test ohne Kosten, q = 4/5: der Wert der Information beträgt 35 GE. Hier stimmen alle Verfahren überein, auch bei fester Testzahl.",
    "Dreistufig (Kapazität, Nachbesserung)": "Der dreistufige Fall (Kapazität, Nachfrage, Nachbesserung) mit der besten ersten Kapazität x₁ = 100: Baum und Szenario-LP geben 132,3125 GE.",
    "Zehn Tests (Knoten gegen Zustände)": "N = 10 Tests nacheinander: 2047 Baum-Knoten gegen 66 Zustände bei gleichen Tests, 161 bei zwei Sorten, 2047 bei Tests mit Primzahl-Quoten.",
    "Exponentialnutzen ρ = 50": "Risikotoleranz ρ = 50 GE, 64 Tangenten, 2000 Szenarien (vorgerechnet): das Tangenten-LP trifft die beste Kapazität der 2000-Punkte-Verteilung bis auf 0,05 GE Mehrverlust.",
    "Zufallsprobleme (1000)": "1000 Zufallsprobleme mit bis zu vier Tests nacheinander: in wie vielen ist das beste Szenario-LP mit fester Testzahl schlechter als das Stoppen nach Beobachtung?",
}

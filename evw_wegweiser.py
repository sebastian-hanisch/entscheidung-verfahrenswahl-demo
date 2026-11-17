"""Wegweiser: welches Verfahren passt zu welcher Aufgabe, und wo hört es auf (Wahrheitstabelle, Belegzeilen, Gegenfälle).

Formfrage zuerst: zuerst zählt die Form des Problems (Informationsstruktur, Nutzen, Zustandsstruktur, Zufallsquelle, Gilt der Erwartungsnutzen), nicht die Größe;
eine Größe (Zahl der Tests, der Knoten) kippt die Empfehlung nie, sie steht nur als Zahl in der Belegzeile. Jede Ausgabe nennt ihre Voraussetzung und eine Belegzeile
aus den Messungen dieser Demo; wo nichts gemessen ist, steht „nicht gemessen“ und keine Empfehlung. Der Wegweiser ist eine Zusammenfassung von Messungen, kein Beweis
einer Regel für andere Probleme. Er heißt hier Wegweiser, nicht „Entscheidungsbaum“: dieses Wort ist in dieser Demo das Rechenverfahren (Ereignisknoten) und
in anderen Demos das Verfahren CART."""

from dataclasses import dataclass

INFO = ("keine Beobachtung vor der Entscheidung", "feste Zahl von Tests, vorab festgelegt", "nach jedem Test stoppen oder weiter testen")
NUTZEN = ("Erwartungswert", "Exponentialnutzen (Risikoaversion)", "anderer Nutzen")
ZUSTAND = ("ja: gleiche Zustände fallen zusammen", "nein: die Quotenverhältnisse der Tests sind multiplikativ unabhängig")
QUELLE = ("diskret gegeben", "Stichprobe aus unbekannter Verteilung")
EU = ("ja: der Erwartungsnutzen gilt", "nein: Allais-Struktur der Wahlen")

ART_VERFAHREN, ART_KEINES, ART_NICHT_GEMESSEN = "verfahren", "keines", "nicht gemessen"


@dataclass(frozen=True)
class Empfehlung:
    art: str             # "verfahren", "keines" (keines der drei Verfahren) oder "nicht gemessen" (keine Empfehlung)
    verfahren: str       # Name des Verfahrens bzw. der Hinweis
    voraussetzung: str
    beleg_id: str        # Schlüssel in BELEGE
    gegenfall: str


def _pct(x):
    return f"{100 * x:.1f}".replace(".", ",") + " %"


def _wechsel(f):
    w = f["wechsel"]["rho"]["50"]
    return w["wechsel"] / f["wechsel"]["nicht_dominiert"]


def beleg_text(beleg_id, f):
    """Belegzeile als Text; die Zahlen stammen aus den eingefrorenen Messreihen `f` (evw_frozen.json) und aus den exakten Rechnungen der Demo."""
    z = f["zufall"]["20261120"]
    seeds = f["zufall"]
    besser = [v["adaptiv_besser"] for v in seeds.values()]
    tang = [r for r in f["tangenten"]["zeilen"] if r["rho"] == 50.0]
    t64 = next(r for r in tang if r["m"] == 64)
    t4 = next(r for r in tang if r["m"] == 4)
    texte = {
        "B_ALLAIS": "Verweis, hier nicht gemessen: Rollback und Szenario-LP rechnen mit Erwartungsnutzen. Gilt er nicht, hilft kein Verfahren dieser Demo; die Nachbar-Demo "
                    "`erwartungsnutzen-grenze-demo` zeigt, an welchen Wahlen das liegt.",
        "B_STICHPROBE": "Im Ein-Standort-Fall gibt das Szenario-LP auf 2000 Szenarien denselben Wert wie die Rechnung über alle Szenarien (`tests/test_oracle_exponential.py`). "
                        "Den Fehler, der aus der Stichprobe selbst kommt, misst diese Demo nicht: Verweis `stichprobenbaum-demo`.",
        "B_ANDERER": "Nicht gemessen: Diese Demo rechnet nur mit dem Erwartungswert und dem Exponentialnutzen. Keine Empfehlung.",
        "B_EXP": (f"Das Szenario-LP mit Erwartungswert wählt bei ρ = 50 in {_pct(_wechsel(f))} der nicht dominierten Zufallsprobleme eine andere Alternative als das Sicherheitsäquivalent "
                  f"(Zufalls-Seed {f['wechsel']['seed']}, {f['wechsel']['m']} Probleme, ein Schritt ohne Test). Das Tangenten-LP nähert den Exponentialnutzen: "
                  f"Mehrverlust {t64['mehrverlust']:.2f} GE bei 64 Tangenten, {t4['mehrverlust']:.1f} GE bei 4 (Ein-Standort-Fall, ρ = 50, 2000 Szenarien, nicht monoton in der Zahl der Tangenten)."
                  .replace(".", ",")),
        "B_KEINE": "Kapazitätsfall ohne Test: Baum und Szenario-LP geben beide 50 GE; dreistufig geben der Baum über alle Knickstellen und das Szenario-LP beide 2117/16 = 132,3125 (`tests/test_oracle_szenario_lp.py`).",
        "B_FEST": (f"Bei fester Testzahl gibt das Szenario-LP denselben Wert wie die Rechnung über die Verteilung der Glaubenszustände: "
                   f"{z['lp_abweichungen']} Abweichungen in {z['lp_geprueft']} LPs ({z['m']} Zufallsprobleme, Zufalls-Seed 20261120)."),
        "B_STOPPEN": (f"Darf nach jedem Test gestoppt werden, ist das beste Szenario-LP mit fester Testzahl in {z['adaptiv_besser']} von {z['m']} Zufallsproblemen schlechter als der Baum "
                      f"({min(besser)} bis {max(besser)} über vier Zufalls-Seeds); das gemischt-ganzzahlige Modell gibt in {z['mip_geprueft'] - z['mip_abweichungen']} von {z['mip_geprueft']} geprüften Problemen den Wert des Baums."),
        "B_ZUSTAND_JA": "Bei N = 14 gleichen Tests hat die Zustandsrechnung 120 Zustände statt 32 767 Knoten (exakt gezählt, `tests/test_zustaende.py`).",
        "B_ZUSTAND_NEIN": "Bei N = 14 Tests mit multiplikativ unabhängigen Quotenverhältnissen hat die Zustandsrechnung 32 767 Zustände, so viele wie der Baum Knoten (Konstruktion mit Primzahl-Quoten, `tests/test_zustaende.py`); "
                          "bei bloß verschiedenen Quotenverhältnissen können Folgen zusammenfallen (Quotenverhältnisse 2, 3, 6: bei N = 3 sind es 14 statt 15 Zustände).",
    }
    return texte[beleg_id]


BELEG_IDS = ("B_ALLAIS", "B_STICHPROBE", "B_ANDERER", "B_EXP", "B_KEINE", "B_FEST", "B_STOPPEN", "B_ZUSTAND_JA", "B_ZUSTAND_NEIN")


def wegweiser(info, nutzen, zustand, quelle, eu):
    """Empfehlung zu den fünf Merkmalen (jeweils ein Eintrag aus INFO, NUTZEN, ZUSTAND, QUELLE, EU). Die Größe der Aufgabe ist absichtlich kein Eingang."""
    if eu == EU[1]:
        return Empfehlung(ART_KEINES, "keines der drei Verfahren", "Rollback und Szenario-LP setzen den Erwartungsnutzen voraus.", "B_ALLAIS",
                          "Kein Gegenfall gemessen; ohne Erwartungsnutzen ist die Frage nach dem Rechenverfahren nicht die erste Frage.")
    if quelle == QUELLE[1]:
        return Empfehlung(ART_VERFAHREN, "Szenario-LP auf der Stichprobe (derselbe Wert wie die Rechnung über die Szenarien)", "Die Szenarien sind eine Stichprobe; der Wert gilt für die Stichprobe, nicht für die Verteilung.",
                          "B_STICHPROBE", "Stichprobenfehler und Optimismus des Besten sind hier nicht gemessen (Verweis `stichprobenbaum-demo`).")
    if nutzen == NUTZEN[2]:
        return Empfehlung(ART_NICHT_GEMESSEN, "nicht gemessen", "Gemessen sind nur Erwartungswert und Exponentialnutzen.", "B_ANDERER", "—")
    if nutzen == NUTZEN[1]:
        return Empfehlung(ART_VERFAHREN, "Baum oder Zustandsrechnung mit Sicherheitsäquivalent im Knoten; das Szenario-LP nur stückweise linear",
                          "Nur beim Exponentialnutzen (konstante Risikoaversion) oder bei linearem Nutzen ist das Sicherheitsäquivalent im Knoten zulässig (Delta-Eigenschaft: Theorie, hier nicht gemessen); das LP des Erwartungswerts wählt in einem Teil der Probleme anders (Belegzeile). Das Stoppen nach Beobachtung mit Exponentialnutzen ist hier nicht gemessen.",
                          "B_EXP", "Bei anderen Nutzenfunktionen (Potenznutzen) hängt die Wahl am Vermögen: hier nicht gemessen.")
    # Erwartungswert, Erwartungsnutzen gilt, diskrete Verteilung
    if info == INFO[0]:
        return Empfehlung(ART_VERFAHREN, "Rollback im Baum (Entscheidungsbaum mit Ereignisknoten) oder Szenario-LP: derselbe Wert",
                          "Feste Informationsstruktur (hier keine Beobachtung) und Erwartungswert.", "B_KEINE",
                          "Der Satz „Baum gleich LP, also egal“ gilt bei fester Informationsstruktur und Erwartungswert; zwei Fälle, in denen er endet, sind das Stoppen nach Beobachtung und der Exponentialnutzen.")
    if info == INFO[1]:
        return Empfehlung(ART_VERFAHREN, "Szenario-LP je feste Testzahl oder Baum ohne Stoppen: derselbe Wert",
                          "Die Zahl der Tests steht vorab fest, es wird nach keinem Ergebnis gestoppt; Erwartungswert.", "B_FEST",
                          "Sobald nach einem Ergebnis gestoppt werden darf, reicht das LP je Testzahl nicht mehr.")
    ja = zustand == ZUSTAND[0]
    return Empfehlung(ART_VERFAHREN,
                      "Baum, Zustandsrechnung oder gemischt-ganzzahliges Modell (nicht das Szenario-LP je Testzahl)" + (": die Zustandsrechnung spart, soweit Zustände zusammenfallen" if ja else ": die Zustandsrechnung spart nichts"),
                      "Stoppen nach Beobachtung ist erlaubt; Erwartungswert, diskrete Verteilung" + ("; gleiche Zustände fallen zusammen" if ja else "; die Zustände fallen nicht zusammen") + ".",
                      "B_STOPPEN",
                      "Eine Vorsichtsregel, kein Aufwandsurteil: in vielen Zufallsproblemen hätte ein einfacheres Verfahren genügt (siehe die Prüfung der Wahlregel).")


def belegzeilen(e, f):
    """Alle Belegzeilen einer Empfehlung: die eigene, und bei „Stoppen“ die Zustandszeile (ja oder nein)."""
    zeilen = [beleg_text(e.beleg_id, f)]
    if e.beleg_id == "B_STOPPEN":
        zeilen.append(beleg_text("B_ZUSTAND_JA" if "spart nichts" not in e.verfahren else "B_ZUSTAND_NEIN", f))
    return zeilen


def alle_kombinationen():
    """Alle Kombinationen der fünf Merkmale (3 · 3 · 2 · 2 · 2 = 72) mit ihrer Empfehlung: die Wahrheitstabelle."""
    return [((i, n, z, q, e), wegweiser(i, n, z, q, e)) for i in INFO for n in NUTZEN for z in ZUSTAND for q in QUELLE for e in EU]

"""Kennzahlen, Verdikt-Logik und Experimente. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text).

Zähleinheiten: Probleme (Zufallsprobleme), Knoten (Historienknoten des Baums), Zustände (Paare aus Tiefe und Glaubenszustand), Variablen und Gleichungen der LPs.
Alle Zahlen gelten für den Kapazitätsfall mit zwei Alternativen und zwei Zuständen (ausgedachte Auszahlungen, ein Modell, kein Fall)."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import evw_constants as C
import evw_exponential as E
import evw_kern as K
import evw_scenario as S
import evw_wegweiser as W

B0 = Fraction(1, 2)
KURZ_ART = {K.TESTARTEN[0]: "gleiche Tests", K.TESTARTEN[1]: "zwei Sorten abwechselnd", K.TESTARTEN[2]: "Tests mit Primzahl-Quoten"}


# ---------------------------------------------------------------------------------------------------------------------
# Darstellung von Zahlen (deutsches Dezimalkomma)
# ---------------------------------------------------------------------------------------------------------------------
def de(x, stellen=1):
    """Dezimalzahl mit Komma und fester Stellenzahl."""
    return f"{float(x):.{stellen}f}".replace(".", ",")


def de_kurz(x, stellen=4):
    """Dezimalzahl mit Komma, ohne überflüssige Nullen am Ende."""
    s = f"{float(x):.{stellen}f}"
    s = s.rstrip("0").rstrip(".") if "." in s else s
    return "0" if s in ("-0", "") else s.replace(".", ",")


def pct(x, stellen=1):
    """Anteil (0 bis 1) als Prozentangabe mit Komma."""
    return de(100 * float(x), stellen) + " %"


def wert_text(x):
    """Wert in GE: Dezimalzahl, bei einem Bruch mit kleinem Nenner zusätzlich der Bruch."""
    if isinstance(x, Fraction):
        s = de_kurz(x)
        return s if x.denominator == 1 or x.denominator > 100000 else f"{s} ({x.numerator}/{x.denominator})"
    return de_kurz(x)


def tausender(n):
    """Ganzzahl mit Leerzeichen als Tausendertrenner (geschütztes Leerzeichen)."""
    return f"{int(n):,}".replace(",", " ")


# ---------------------------------------------------------------------------------------------------------------------
# Experiment 1: dieselbe Entscheidung in vier Verfahren
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Verfahren1:
    n_max: int
    q: Fraction
    c: Fraction
    stoppen: bool
    baum_wert: object
    strategie: object
    baum_knoten: int
    baum_innere: int
    dp_wert: object
    dp_zustaende: int
    fest_werte: list
    bestes_n: int
    bester_fest: object
    lp: tuple                # (Wert, Variablen, Gleichungen) für das beste feste n
    mip: object              # (Wert, Variablen, Binärvariablen, Zeilen) oder None
    abstand: object          # Baum minus bester fester Wert
    ohne_test: dict
    evsi_ein_test: object


def als_bruch(x, grenze=1000):
    return Fraction(str(x)).limit_denominator(grenze)


def analyse_tests(n_max, q20, c, stoppen):
    """Rechnet den Kapazitätsfall mit bis zu n_max Tests (Sensitivität = Spezifität = q20/20, Kosten c GE) in den Verfahren. Baum und Zustandsrechnung in Brüchen."""
    q, cf = Fraction(q20, 20), als_bruch(c)
    tests = K.tests_gleich(q, n_max)
    fest = [K.fest_wert(B0, K.A_CAP, cf, tests, n) for n in range(n_max + 1)]
    bestes_n = max(range(n_max + 1), key=lambda n: (fest[n], -n))      # bei Gleichstand die kleinere Zahl der Tests
    bester = fest[bestes_n]
    lp = K.fest_lp(B0, K.A_CAP, cf, tests, bestes_n)
    if stoppen:
        z = K.Zaehler()
        wert, strategie = K.baum(B0, K.A_CAP, cf, tests, 0, z)
        dp_wert, _, dp_z = K.dp(B0, K.A_CAP, cf, tests)
        mip = K.adaptiv_mip(B0, K.A_CAP, cf, tests, n_max) if n_max <= C.MIP_N_MAX else None
        knoten, innere = z.knoten, z.innere
    else:
        wert = K.baum_ohne_stoppen(B0, K.A_CAP, cf, tests[:bestes_n])
        strategie = None
        dp_wert = bester
        dp_z = sum(K.zaehle_zustaende(B0, tests[:bestes_n]))
        mip = None
        knoten, innere = K.baum_knoten(bestes_n)
    return Verfahren1(n_max, q, cf, stoppen, wert, strategie, knoten, innere, dp_wert, dp_z, fest, bestes_n, bester, lp, mip, wert - bester,
                      K.ohne_test(B0, K.A_CAP), K.evsi_ein_test(B0, K.A_CAP, q) if n_max >= 0 else None)


def zeilen_verfahren(v: Verfahren1):
    """Tabelle „Wert, Größe, Abstand zum Baum“ als Liste von Wörterbüchern (Texte, damit überall dasselbe Dezimalzeichen steht)."""
    kopf_baum = "Baum (Entscheidungsbaum mit Ereignisknoten)" + ("" if v.stoppen else f", ohne Stoppen, n = {v.bestes_n}")
    zeilen = [
        {"Verfahren": kopf_baum, "Wert (GE)": wert_text(v.baum_wert), "Größe": f"{v.baum_knoten} Knoten, davon {v.baum_innere} innere" if v.stoppen else f"{v.baum_knoten} Knoten",
         "Abstand zum Baum (GE)": "0"},
        {"Verfahren": "Zustandsrechnung (Tiefe, Glaubenszustand)", "Wert (GE)": wert_text(v.dp_wert), "Größe": f"{v.dp_zustaende} Zustände",
         "Abstand zum Baum (GE)": de_kurz(v.dp_wert - v.baum_wert, 6)},
        {"Verfahren": f"Szenario-LP, feste Testzahl n = {v.bestes_n} (bestes n)", "Wert (GE)": de_kurz(v.lp[0], 6), "Größe": f"{v.lp[1]} Variablen, {v.lp[2]} Gleichungen",
         "Abstand zum Baum (GE)": de_kurz(v.lp[0] - float(v.baum_wert), 6)},
    ]
    if v.stoppen:
        if v.mip is not None:
            zeilen.append({"Verfahren": "gemischt-ganzzahliges Modell (MIP)", "Wert (GE)": de_kurz(v.mip[0], 6), "Größe": f"{v.mip[1]} Variablen, davon {v.mip[2]} binär, {v.mip[3]} Zeilen",
                           "Abstand zum Baum (GE)": de_kurz(v.mip[0] - float(v.baum_wert), 6)})
        else:
            zeilen.append({"Verfahren": "gemischt-ganzzahliges Modell (MIP)", "Wert (GE)": "nicht gerechnet", "Größe": f"N über {C.MIP_N_MAX}: Laufzeit über 2 Sekunden", "Abstand zum Baum (GE)": "—"})
    else:
        zeilen.append({"Verfahren": "gemischt-ganzzahliges Modell (MIP)", "Wert (GE)": "—", "Größe": "nur für das Stoppen nach Beobachtung nötig", "Abstand zum Baum (GE)": "—"})
    return zeilen


def verdict_verfahren(v: Verfahren1):
    """Verdikt zu Experiment 1 (Kapazitätsfall, ausgedachte Zahlen)."""
    ev = v.ohne_test
    kennzahl = (f"EVPI {wert_text(ev['evpi'])} GE = {pct(ev['evpi_von_ws'])} von WS "
                f"(RP = {wert_text(ev['rp'])} GE, WS = {wert_text(ev['ws'])} GE; Kapazitätsfall, diese Rechnung)")
    if not v.stoppen:
        return "success", (f"Ohne Stoppen geben Baum, Zustandsrechnung und Szenario-LP denselben Wert: **{wert_text(v.baum_wert)} GE** bei n = {v.bestes_n} Tests "
                           f"(Abstand höchstens 1e-9). {kennzahl}.")
    mip_text = f", das gemischt-ganzzahlige Modell {de_kurz(v.mip[0], 3)}" if v.mip is not None else ""
    if v.abstand > Fraction(1, 10 ** 9):
        return "success", (f"Baum und Zustandsrechnung geben **{wert_text(v.baum_wert)} GE**{mip_text}; bei fester Testzahl gibt das Szenario-LP höchstens {de_kurz(v.bester_fest, 4)} GE (n = {v.bestes_n}): "
                           f"**Abstand {de_kurz(v.abstand, 4)} GE** zum Stoppen nach Beobachtung. {kennzahl}.")
    return "info", (f"Hier reicht eine feste Testzahl: Baum, Zustandsrechnung{mip_text.replace(', das', ' und das') if mip_text else ''} und das beste Szenario-LP (n = {v.bestes_n}) "
                    f"geben alle {wert_text(v.baum_wert)} GE (Abstand höchstens 1e-9). {kennzahl}.")


# ---------------------------------------------------------------------------------------------------------------------
# Der Baum als Diagramm (bis N = 3)
# ---------------------------------------------------------------------------------------------------------------------
def baum_info(b0, alts, c, tests):
    """Alle Historienknoten mit Glaubenszustand, Stopp-Wert, Weiter-Wert, Wert und der Angabe, ob die optimale Strategie den Knoten erreicht und dort stoppt."""
    def rec(b, j, pfad):
        s = K.stop_wert(b, alts)
        k = {"pfad": pfad, "tiefe": j, "b": b, "stopp": s, "wahl": K.stop_wahl(b, alts), "weiter": None, "kinder": None}
        if j >= len(tests):
            k.update(wert=s, stoppt=True)
            return k
        (p1, b1), (p2, b2) = K.update(b, tests[j], True), K.update(b, tests[j], False)
        k1, k2 = rec(b1, j + 1, pfad + "+"), rec(b2, j + 1, pfad + "−")
        weiter = -c + p1 * k1["wert"] + p2 * k2["wert"]
        k.update(weiter=weiter, wert=max(s, weiter), stoppt=not weiter > s, kinder=(k1, k2))
        return k

    wurzel = rec(b0, 0, "")
    flach = []

    def markiere(k, erreichbar):
        k["erreichbar"] = erreichbar
        flach.append(k)
        if k["kinder"]:
            for kind in k["kinder"]:
                markiere(kind, erreichbar and not k["stoppt"])

    markiere(wurzel, True)
    return flach


# ---------------------------------------------------------------------------------------------------------------------
# Dreistufiger Fall
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Dreistufig:
    x1: int
    wert_x1: Fraction
    optimum: Fraction
    x1_optimum: Fraction
    knickstellen: int
    lp_wert: float
    lp_x1: float
    lp_variablen: int
    lp_szenarien: int
    raster: list


def analyse_dreistufig(x1):
    opt, x_opt, kn = K.dreistufig_baum()
    lp_w, lp_x, nv, ns = K.dreistufig_lp()
    raster = [(x, float(K.dreistufig_wert(x))) for x in range(0, 201, 5)]
    return Dreistufig(x1, K.dreistufig_wert(x1), opt, x_opt, kn, lp_w, lp_x, nv, ns, raster)


def verdict_dreistufig(d: Dreistufig):
    gleich = abs(float(d.optimum) - d.lp_wert) < 1e-9
    text = (f"Bei einer ersten Kapazität x₁ = {d.x1} kostet der Baum im Erwartungswert {wert_text(d.wert_x1)} GE. Das Optimum des Baums (alle {d.knickstellen} Knickstellen in Brüchen geprüft) liegt bei "
            f"x₁ = {wert_text(d.x1_optimum)} mit {wert_text(d.optimum)} GE; das Szenario-LP mit nichtantizipativer Nachbesserung ({d.lp_variablen} Variablen, {d.lp_szenarien} Szenarien) gibt "
            f"{de_kurz(d.lp_wert, 6)} GE bei x₁ = {de_kurz(d.lp_x1, 3)}" + (" (Abstand höchstens 1e-9)." if gleich else "."))
    return ("success" if gleich and d.x1_optimum == d.x1 else "info"), text


# ---------------------------------------------------------------------------------------------------------------------
# Experiment 2: Zufallsprobleme (Stoppen nach Beobachtung gegen feste Testzahl)
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class ZufallsLauf:
    seed: int
    m: int
    n_tests: int
    art: str
    nicht_dominiert: int
    test_lohnt: int
    adaptiv_besser: int
    besser_unter_lohnend: int
    dp_abweichungen: int
    luecken: list                 # Lücke (Baum minus bestes festes n) je Problem
    luecken_lohnend: list
    mittel_baum_lohnend: float
    mittel_fest_lohnend: float
    index_max: int                # Problem mit der größten Lücke
    probleme: list = None


def zufall_lauf(seed, m, n_tests, art, q20, c):
    """m Zufallsprobleme aus dem Zufalls-Seed; bis zu n_tests gleiche Tests nacheinander. Baum (Gleitkomma) gegen Zustandsrechnung gegen bestes festes n (Verteilungsrechnung)."""
    probleme = S.ziehe_probleme(seed, m)
    if art == C.ART_FEST:
        probleme = [S.Problem(p.alts, p.b0, q20 / 20.0, float(c)) for p in probleme]
    nd = lohnt = besser = besser_l = dp_abw = 0
    luecken, luecken_l = [], []
    baum_l, fest_l = [], []
    for p in probleme:
        if not S.dominiert(p.alts):
            nd += 1
        tests = K.tests_gleich(p.q, n_tests)
        v_baum, _ = K.baum(p.b0, p.alts, p.c, tests)
        if abs(v_baum - K.dp_wert_float(p.b0, p.alts, p.c, p.q, n_tests)) > 1e-9:
            dp_abw += 1
        bester = max(K.fest_wert(p.b0, p.alts, p.c, tests, n) for n in range(n_tests + 1))
        luecke = v_baum - bester
        luecken.append(luecke)
        if v_baum > K.stop_wert(p.b0, p.alts) + 1e-9:
            lohnt += 1
            luecken_l.append(luecke)
            baum_l.append(v_baum)
            fest_l.append(bester)
            if luecke > 1e-7:
                besser_l += 1
        if luecke > 1e-7:
            besser += 1
    imax = int(np.argmax(luecken)) if luecken else 0
    return ZufallsLauf(seed, m, n_tests, art, nd, lohnt, besser, besser_l, dp_abw, luecken, luecken_l,
                       float(np.mean(baum_l)) if baum_l else 0.0, float(np.mean(fest_l)) if fest_l else 0.0, imax, probleme)


def detail_problem(p: S.Problem, n_tests):
    """Ein Problem in allen Verfahren (live): Baum, bestes festes n, Szenario-LP je n, MIP (bis C.MIP_N_MAX)."""
    tests = K.tests_gleich(p.q, n_tests)
    wert, strat = K.baum(p.b0, p.alts, p.c, tests)
    fest = [K.fest_wert(p.b0, p.alts, p.c, tests, n) for n in range(n_tests + 1)]
    lp = [K.fest_lp(p.b0, p.alts, p.c, tests, n)[0] for n in range(n_tests + 1)]
    mip = K.adaptiv_mip(p.b0, p.alts, p.c, tests, n_tests)[0] if n_tests <= C.MIP_N_MAX else None
    return {"baum": wert, "strategie": strat, "fest": fest, "lp": lp, "mip": mip, "ohne_test": K.stop_wert(p.b0, p.alts)}


def verdict_zufall(z: ZufallsLauf, frozen):
    """Verdikt zu Experiment 2: nennt Probleme, Zufalls-Seed, Zählung, Mittel und die Spanne der vorgerechneten Zufalls-Seeds."""
    besser = [v["adaptiv_besser"] for v in frozen["zufall"].values()]
    anker = (f" Zum Vergleich (vorgerechnet, je 1000 Probleme, N = 4, zufällige Testgüte und Kosten, vier Zufalls-Seeds): {min(besser)} bis {max(besser)} von 1000.")
    if z.test_lohnt == 0:
        return "warning", f"In keinem der {z.m} Zufallsprobleme (Zufalls-Seed {z.seed}) lohnt sich ein Test: es gibt nichts zu stoppen." + anker
    reicht = z.test_lohnt - z.besser_unter_lohnend
    mittel = float(np.mean(z.luecken_lohnend))
    return "success", (f"Von {z.m} Zufallsproblemen (Zufalls-Seed {z.seed}, bis zu {z.n_tests} gleiche Tests nacheinander) ist Stoppen nach Beobachtung in **{z.adaptiv_besser}** ({pct(z.adaptiv_besser / z.m)}) "
                       f"besser als jede feste Testzahl; in {z.test_lohnt} lohnt sich überhaupt ein Test, dort beträgt die Lücke im Mittel {de(mittel, 2)} GE (größte {de(max(z.luecken_lohnend), 2)} GE), "
                       f"in {reicht} davon reicht eine feste Zahl. Baum und Zustandsrechnung weichen in {z.dp_abweichungen} Problemen ab." + anker)


# ---------------------------------------------------------------------------------------------------------------------
# Experiment 3: Knoten gegen Zustände
# ---------------------------------------------------------------------------------------------------------------------
def zustaende_tabellen(n_max=C.E3_N_MAX):
    """Für jede Testart die Zahl der Zustände (Summe über die Tiefen 0..N) und die je Tiefe, für N = 0..n_max (Brüche, exakt gezählt)."""
    aus = {}
    for art in K.TESTARTEN:
        je_tiefe = K.zaehle_zustaende(B0, K.testfolge(art, n_max))
        summe, s = [], 0
        for z in je_tiefe:
            s += z
            summe.append(s)
        aus[art] = {"je_tiefe": je_tiefe, "summe": summe}
    return aus


def glaubenszustaende_je_tiefe(art, n):
    """Die Glaubenszustände (Brüche) je Tiefe 0..n für die Tabelle (nur für kleine n sinnvoll)."""
    tests = K.testfolge(art, n)
    return [[f"{b.numerator}/{b.denominator}" for b in K.glaubenszustaende(B0, tests, t)] for t in range(n + 1)]


def verdict_zustaende(n, art, tab):
    """Verdikt zu Experiment 3 an der gewählten Zahl n der Tests und Testart."""
    knoten = K.baum_knoten(n)[0]
    z = {a: tab[a]["summe"][n] for a in K.TESTARTEN}
    mine = z[art]
    text = (f"N = {n}: {tausender(knoten)} Baum-Knoten; Zustände bei gleichen Tests {tausender(z[K.TESTARTEN[0]])}, bei zwei Sorten abwechselnd {tausender(z[K.TESTARTEN[1]])}, "
            f"bei Tests mit Primzahl-Quoten (multiplikativ unabhängige Quotenverhältnisse) {tausender(z[K.TESTARTEN[2]])} (exakt gezählt, Brüche). Gewählt: {art}, {tausender(mine)} Zustände = {pct(mine / knoten)} der Knoten.")
    return ("info" if mine < knoten else "warning"), text


# ---------------------------------------------------------------------------------------------------------------------
# Experiment 4: Exponentialnutzen im LP
# ---------------------------------------------------------------------------------------------------------------------
def tangenten_ergebnis(rho, m, szen, frozen):
    """Beste Kapazität exakt gegen Tangenten-LP: bei 2000 Szenarien aus der vorgerechneten Tabelle, sonst live (250 oder 500 Szenarien)."""
    if szen == C.E4_SZEN_VORGERECHNET:
        z = next(r for r in frozen["tangenten"]["zeilen"] if int(r["rho"]) == rho and r["m"] == m)
        return {"x_exakt": z["x_exakt"], "ce_exakt": z["ce_exakt"], "x_lp": z["x_lp"], "mehrverlust": z["mehrverlust"], "live": False, "szenarien": szen}
    d = E.nachfrage(szen)
    x_ex, ce_ex = E.beste_kapazitaet_ce(d, float(rho))
    x_lp, nv, nz = E.tangenten_lp(d, float(rho), m, ce_ex)
    return {"x_exakt": x_ex, "ce_exakt": ce_ex, "x_lp": x_lp, "mehrverlust": E.mehrverlust(d, float(rho), x_lp, ce_ex), "live": True, "szenarien": szen, "variablen": nv, "ungleichungen": nz}


def kurve_tangenten(rho, m, ce_ref, halbbreite=None):
    """Daten für das Diagramm: exp((c - cref)/rho) gegen die Tangenten-Hüllkurve, in einem Fenster um cref."""
    halb = halbbreite or max(3.0 * rho, 60.0)
    von, bis = max(E.TANGENTEN_VON, ce_ref - halb), min(E.TANGENTEN_BIS, ce_ref + halb)
    c = np.linspace(von, bis, 200)
    exakt = np.exp((c - ce_ref) / rho)
    stuetz, a, b = E.tangenten(m, rho, ce_ref)
    huelle = np.max(a[:, None] * c[None, :] + b[:, None], axis=0)
    innen = (stuetz >= von) & (stuetz <= bis)
    return c, exakt, huelle, stuetz[innen], np.exp((stuetz[innen] - ce_ref) / rho)


def wechsel_live(seed, m_probleme, rho):
    """Entscheidungswechsel für die vier Standard-Risikotoleranzen und die gewählte (ein Schritt, ohne Test)."""
    probleme = S.ziehe_probleme(seed, m_probleme)
    gezaehlt, erg = E.wechsel_zaehlung(probleme, sorted(set(C.E4_WECHSEL_RHOS) | {float(rho)}))
    return {"nicht_dominiert": gezaehlt, "rho": erg, "m": m_probleme, "seed": seed}


def verdict_exponential(rho, m, t, w, frozen):
    """Verdikt zu Experiment 4: Mehrverlust des Tangenten-LP und Anteil der Probleme, in denen das risikoneutrale LP anders wählt."""
    anteil = w["rho"][float(rho)]["wechsel"] / w["nicht_dominiert"] if w["nicht_dominiert"] else 0.0
    quelle = "vorgerechnet" if not t["live"] else "live gerechnet"
    gl = frozen["wechsel"]
    anker = (f" Vorgerechnet (Zufalls-Seed {gl['seed']}, {tausender(gl['m'])} Probleme): "
             + ", ".join(f"ρ = {r}: {pct(v['wechsel'] / gl['nicht_dominiert'])}" for r, v in gl["rho"].items()) + ".")
    return "info", (f"ρ = {rho} GE: das risikoneutrale LP wählt in **{pct(anteil)}** der {w['nicht_dominiert']} nicht dominierten Zufallsprobleme (Zufalls-Seed {w['seed']}, ein Schritt ohne Test) "
                    f"eine andere Alternative als das Sicherheitsäquivalent. Das Tangenten-LP ({m} Tangenten, {t['szenarien']} Szenarien, {quelle}) wählt die Kapazität {de(t['x_lp'], 1)} statt {de(t['x_exakt'], 1)}: "
                    f"Mehrverlust **{de(t['mehrverlust'], 3)} GE** bei einem Sicherheitsäquivalent von {de(t['ce_exakt'], 1)} GE (Ein-Standort-Fall, Kosten)." + anker)


# ---------------------------------------------------------------------------------------------------------------------
# Experiment 5: Wegweiser, geprüft
# ---------------------------------------------------------------------------------------------------------------------
def wahlregel_pruefung(z: ZufallsLauf):
    """Prüfung der Regel „Stoppen erlaubt: Baum, Zustandsrechnung oder gemischt-ganzzahliges Modell“ an den Zufallsproblemen aus Experiment 2.
    Das Szenario-LP je Testzahl verfehlt den Baumwert in k Problemen; die Regel überempfiehlt dort, wo kein Test lohnt oder eine feste Zahl reicht."""
    return {"m": z.m, "verfehlt": z.adaptiv_besser, "trifft": z.m - z.dp_abweichungen, "ueberempfiehlt": z.m - z.adaptiv_besser,
            "kein_test": z.m - z.test_lohnt, "fest_reicht": z.test_lohnt - z.besser_unter_lohnend, "lohnt": z.test_lohnt}


def verdict_wegweiser(e: W.Empfehlung, p):
    """Verdikt zur Prüfung der Wahlregel (nur für die Empfehlung „Stoppen“ definiert)."""
    if e.beleg_id != "B_STOPPEN":
        return "info", "Die Prüfung der Wahlregel an den Zufallsproblemen gilt für die Frage „Stoppen nach Beobachtung“; für diese Auswahl ist sie nicht gerechnet."
    return "success", (f"Regel „Stoppen erlaubt: Baum, Zustandsrechnung oder gemischt-ganzzahliges Modell“, geprüft an {p['m']} Zufallsproblemen aus Experiment 2: das Szenario-LP je Testzahl verfehlt den Baumwert in "
                       f"**{p['verfehlt']}**, die Zustandsrechnung trifft ihn in {p['trifft']}. Die Regel überempfiehlt in den {p['ueberempfiehlt']} anderen ({p['kein_test']} ohne lohnenden Test, "
                       f"{p['fest_reicht']} mit ausreichender fester Testzahl): eine Vorsichtsregel, kein Aufwandsurteil für den Einzelfall.")


# ---------------------------------------------------------------------------------------------------------------------
# Kernfragen (live, aus den Experimenten)
# ---------------------------------------------------------------------------------------------------------------------
def kernfragen(v: Verfahren1, z: ZufallsLauf, n3, art3, tab):
    """Drei Kernfragen mit je einer Antwort (Art, Text) aus den laufenden Rechnungen."""
    knoten = K.baum_knoten(n3)[0]
    mine = tab[art3]["summe"][n3]
    mip = " und gemischt-ganzzahliges Modell" if v.mip is not None else ""
    if v.stoppen and v.abstand > Fraction(1, 10 ** 9):
        zusatz = f"; das beste Szenario-LP mit fester Testzahl bleibt {de_kurz(v.abstand, 4)} GE darunter."
    elif v.stoppen:
        zusatz = "; hier reicht auch eine feste Testzahl, das Szenario-LP trifft ihn."
    else:
        zusatz = "; auch das Szenario-LP trifft ihn."
    frage1 = (f"**Ja, im Wert, aber nicht bei jeder Informationsstruktur.** Bei diesen Einstellungen (N = {v.n_max}, q = {v.q}, c = {wert_text(v.c)} GE) geben Baum, Zustandsrechnung{mip} "
              f"{wert_text(v.baum_wert)} GE" + zusatz)
    frage2 = (f"**Beim Stoppen nach Beobachtung.** In {z.adaptiv_besser} von {z.m} Zufallsproblemen (Zufalls-Seed {z.seed}) ist das beste Szenario-LP mit fester Testzahl schlechter als der Baum; "
              f"beim Exponentialnutzen wählt das risikoneutrale LP in einem Teil der Probleme anders (Experiment 4).")
    frage3 = f"**Nur, wo Zustände zusammenfallen.** Bei N = {n3} Tests ({KURZ_ART[art3]}) hat die Zustandsrechnung {tausender(mine)} Zustände gegen {tausender(knoten)} Baum-Knoten."
    return [("info", frage1), ("info", frage2), ("info", frage3)]

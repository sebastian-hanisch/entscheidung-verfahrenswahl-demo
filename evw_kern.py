"""Kern: dieselbe Entscheidung in vier Verfahren (Entscheidungsbaum mit Ereignisknoten, Zustandsrechnung, Szenario-LP, gemischt-ganzzahliges Modell).

Modell (ausgedachte Zahlen, kein Fall): Alternative G (groß bauen) zahlt bei hoher Nachfrage +200, bei niedriger -100 Geldeinheiten (GE); Alternative K
(klein bauen oder mieten) zahlt sicher +30 GE. P(hoch) = 1/2. Ein Test hat Sensitivität und Spezifität q und kostet c GE; bis zu N Tests nacheinander,
nach jedem Ergebnis darf gestoppt werden. Der **Glaubenszustand** b = P(hoch) wird nach der Regel von Bayes fortgeschrieben.

Alle Rechnungen laufen mit `fractions.Fraction` (exakt) oder mit Gleitkomma; die Funktionen sind für beides geschrieben. Zufall kommt hier nicht vor.
Eigene Rechnung nach der Lehrbuchlage (Rollback: Raiffa und Schlaifer 1961; Zustandsrechnung: Bellman; Szenario-LP: Birge und Louveaux).
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import optimize, sparse

ALT_G, ALT_K = 0, 1
ALT_NAMEN = ("G (groß bauen)", "K (klein bauen oder mieten)")
ALT_KURZ = ("G", "K")
A_CAP = ((200, -100), (30, 30))  # je Alternative (Auszahlung bei hoher, bei niedriger Nachfrage)
MIP_GROSSM = 500.0
MIP_LUECKE = 1e-9  # Toleranz der Software (HiGHS: Standard 1e-4): ohne sie weicht das gemischt-ganzzahlige Modell in einzelnen Problemen minimal ab


def ew(alt, b):
    """Erwartungswert einer Alternative bei Glaubenszustand b = P(hoch)."""
    return b * alt[0] + (1 - b) * alt[1]


def stop_wert(b, alts=A_CAP):
    """Wert, jetzt zu entscheiden: der beste Erwartungswert (GE)."""
    return max(ew(a, b) for a in alts)


def stop_wahl(b, alts=A_CAP):
    """Index der besten Alternative; bei Gleichstand die kleinere Nummer (G vor K)."""
    return ALT_K if ew(alts[ALT_K], b) > ew(alts[ALT_G], b) else ALT_G


def update(b, test, plus):
    """(Wahrscheinlichkeit des Ergebnisses, neuer Glaubenszustand); test = (Sensitivität, Spezifität).
    Hat das Ergebnis die Wahrscheinlichkeit 0, fällt der Zweig weg: der Glaubenszustand bleibt, damit nicht durch 0 geteilt wird."""
    se, sp = test
    if plus:
        pr = b * se + (1 - b) * (1 - sp)
        return (pr, b) if pr == 0 else (pr, b * se / pr)
    pr = b * (1 - se) + (1 - b) * sp
    return (pr, b) if pr == 0 else (pr, b * (1 - se) / pr)


def tests_gleich(q, n):
    """n gleiche Tests mit Sensitivität = Spezifität = q."""
    return [(q, q)] * n


# ---------------------------------------------------------------------------------------------------------------------
# Verfahren 1: Entscheidungsbaum mit Ereignisknoten (Rollback ohne Gedächtnis: jeder Knoten wird einzeln gerechnet)
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Zaehler:
    knoten: int = 0     # Historienknoten: ein Knoten je Folge von Testergebnissen der Länge höchstens N
    innere: int = 0     # davon Knoten, an denen noch ein Test möglich ist (Ereignisknoten folgen)


def baum(b, alts, c, tests, j=0, z=None):
    """Rollback im ausgeschriebenen Baum. Rückgabe (Wert, Strategie); Strategie = ("stop", Alternative) oder ("test", (Strategie nach +, Strategie nach -)).
    Bei Gleichstand wird gestoppt (festgelegt, sonst sind Strategien nicht vergleichbar)."""
    if z is not None:
        z.knoten += 1
    s = stop_wert(b, alts)
    if j >= len(tests):
        return s, ("stop", stop_wahl(b, alts))
    if z is not None:
        z.innere += 1
    (p1, b1), (p2, b2) = update(b, tests[j], True), update(b, tests[j], False)
    v1, s1 = baum(b1, alts, c, tests, j + 1, z)
    v2, s2 = baum(b2, alts, c, tests, j + 1, z)
    weiter = -c + p1 * v1 + p2 * v2
    if weiter > s:
        return weiter, ("test", (s1, s2))
    return s, ("stop", stop_wahl(b, alts))


def baum_knoten(N):
    """Zahl der Historienknoten bei bis zu N Tests nacheinander: 2^(N+1) - 1, davon 2^N - 1 innere."""
    return 2 ** (N + 1) - 1, 2 ** N - 1


def baum_ohne_stoppen(b, alts, c, tests, j=0):
    """Rollback ohne die Möglichkeit zu stoppen: alle Tests werden gemacht (feste Testzahl len(tests)). Rückgabe nur der Wert."""
    if j >= len(tests):
        return stop_wert(b, alts)
    (p1, b1), (p2, b2) = update(b, tests[j], True), update(b, tests[j], False)
    return -c + p1 * baum_ohne_stoppen(b1, alts, c, tests, j + 1) + p2 * baum_ohne_stoppen(b2, alts, c, tests, j + 1)


# ---------------------------------------------------------------------------------------------------------------------
# Verfahren 2: Zustandsrechnung (dieselbe Rückwärtsrechnung mit Gedächtnis auf dem Paar (Tiefe, Glaubenszustand))
# ---------------------------------------------------------------------------------------------------------------------
def dp(b0, alts, c, tests):
    """Rückgabe (Wert, Strategie, Zahl der Zustände). Ein Zustand ist ein Paar (Tiefe, Glaubenszustand); zwei Ergebnisfolgen mit demselben Bruch sind ein Zustand."""
    memo = {}

    def rec(b, j):
        key = (j, b)
        if key in memo:
            return memo[key]
        s = stop_wert(b, alts)
        if j >= len(tests):
            memo[key] = (s, ("stop", stop_wahl(b, alts)))
            return memo[key]
        (p1, b1), (p2, b2) = update(b, tests[j], True), update(b, tests[j], False)
        v1, s1 = rec(b1, j + 1)
        v2, s2 = rec(b2, j + 1)
        weiter = -c + p1 * v1 + p2 * v2
        memo[key] = (weiter, ("test", (s1, s2))) if weiter > s else (s, ("stop", stop_wahl(b, alts)))
        return memo[key]

    wert, strategie = rec(b0, 0)
    return wert, strategie, len(memo)


def dp_wert_float(b0, alts, c, q, N):
    """Wert der Zustandsrechnung für N gleiche Tests in Gleitkomma, schnell (nur der Wert, für die vielen Zufallsprobleme)."""
    se = sp = q
    stufe = {b0: 1.0}
    folge = [stufe]
    for _ in range(N):
        neu = {}
        for b in stufe:
            for plus in (True, False):
                pr, nb = update(b, (se, sp), plus)
                if pr != 0:
                    neu[nb] = 0.0
        folge.append(neu)
        stufe = neu
    werte = {b: stop_wert(b, alts) for b in folge[N]}
    for j in range(N - 1, -1, -1):
        neu_werte = {}
        for b in folge[j]:
            (p1, b1), (p2, b2) = update(b, (se, sp), True), update(b, (se, sp), False)
            weiter = -c + p1 * werte[b1] + p2 * werte[b2]
            neu_werte[b] = max(stop_wert(b, alts), weiter)
        werte = neu_werte
    return werte[b0]


def zaehle_zustaende(b0, tests):
    """Verschiedene Glaubenszustände je Tiefe 0..N (Brüche, exakt gezählt); Rückgabe die Liste der Zahlen je Tiefe. Die Summe ist die Zahl der Zustände der Zustandsrechnung."""
    stufe = {b0}
    je_tiefe = [1]
    for test in tests:
        neu = set()
        for b in stufe:
            for plus in (True, False):
                pr, nb = update(b, test, plus)
                if pr != 0:
                    neu.add(nb)
        stufe = neu
        je_tiefe.append(len(stufe))
    return je_tiefe


def glaubenszustaende(b0, tests, tiefe):
    """Die verschiedenen Glaubenszustände (Brüche) in der gegebenen Tiefe, aufsteigend sortiert."""
    stufe = {b0}
    for test in tests[:tiefe]:
        neu = set()
        for b in stufe:
            for plus in (True, False):
                pr, nb = update(b, test, plus)
                if pr != 0:
                    neu.add(nb)
        stufe = neu
    return sorted(stufe)


# Testarten für den Vergleich Knoten gegen Zustände (Experiment 3)
PRIMZAHLEN = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
TESTARTEN = ("gleiche Tests", "zwei Sorten abwechselnd", "paarweise verschieden (Primzahl-Quoten)")


def testfolge(art, N):
    """Folge von N Tests: lauter gleiche (q = 4/5), zwei Sorten (4/5 und 3/5 abwechselnd) oder paarweise verschiedene Quotenverhältnisse.
    Bei der dritten Art hat der Test Nr. i das Quotenverhältnis q/(1 - q) gleich der i-ten Primzahl p_i (q = p_i/(1 + p_i)); die Produkte der p_i^(+1) und p_i^(-1)
    sind nach der Primfaktorzerlegung paarweise verschieden: eine Konstruktion, keine zufällige Wahl der Tests."""
    if art == TESTARTEN[0]:
        return tests_gleich(Fraction(4, 5), N)
    if art == TESTARTEN[1]:
        return [((Fraction(4, 5),) * 2 if j % 2 == 0 else (Fraction(3, 5),) * 2) for j in range(N)]
    if art == TESTARTEN[2]:
        return [(Fraction(PRIMZAHLEN[i], 1 + PRIMZAHLEN[i]),) * 2 for i in range(N)]
    raise ValueError(art)


def zustaende_formel(art, N):
    """Geschlossene Formel der Zustandszahl (Summe über die Tiefen 0..N): gleiche Tests (N+1)(N+2)/2; zwei Sorten Summe über j von (ceil(j/2)+1)(floor(j/2)+1); verschieden 2^(N+1)-1."""
    if art == TESTARTEN[0]:
        return (N + 1) * (N + 2) // 2
    if art == TESTARTEN[1]:
        return sum(((j + 1) // 2 + 1) * (j // 2 + 1) for j in range(N + 1))
    return 2 ** (N + 1) - 1


# ---------------------------------------------------------------------------------------------------------------------
# Feste Testzahl: Verteilung der Glaubenszustände (Rechnung) und Szenario-LP mit Nichtantizipativität
# ---------------------------------------------------------------------------------------------------------------------
def fest_wert(b0, alts, c, tests, n):
    """Wert bei fester Zahl n der Tests (die ersten n der Folge), danach wird entschieden; Stoppen ist nicht erlaubt. Verteilung der Glaubenszustände, gleiche Zustände zusammengelegt."""
    eins = Fraction(1) if isinstance(b0, Fraction) else 1.0
    verteilung = {b0: eins}
    for j in range(n):
        neu = {}
        for b, w in verteilung.items():
            for plus in (True, False):
                pr, nb = update(b, tests[j], plus)
                if pr == 0:
                    continue
                neu[nb] = neu.get(nb, 0) + w * pr
        verteilung = neu
    return -c * n + sum(w * stop_wert(b, alts) for b, w in verteilung.items())


def fest_lp(b0, alts, c, tests, n):
    """Szenario-LP für eine feste Zahl n der Tests. Szenarien sind Paare (Zustand der Nachfrage, Folge der n Testergebnisse); je Szenario ein Anteil x_s der Alternative G,
    nichtantizipativ: die beiden Szenarien mit gleicher Ergebnisfolge haben denselben Anteil (Gleichung). Zwei Alternativen und zwei Zustände.
    Rückgabe (Wert, Zahl der Variablen, Zahl der Gleichungen). Gleichgesetzt wird der Wert mit der Rechnung über die Verteilung (fest_wert)."""
    b0 = float(b0)
    h0, l0 = alts[ALT_G]
    h1, l1 = alts[ALT_K]
    folgen = 2 ** n
    nz = 2 * folgen
    cvec = np.zeros(nz)
    for fi in range(folgen):
        ph, pl = b0, 1.0 - b0
        for i in range(n):
            se, sp = float(tests[i][0]), float(tests[i][1])
            plus = bool((fi >> i) & 1)
            ph *= se if plus else 1.0 - se
            pl *= (1.0 - sp) if plus else sp
        cvec[2 * fi] = -ph * (h0 - h1)       # Gewinn von G gegen K bei hoher Nachfrage
        cvec[2 * fi + 1] = -pl * (l0 - l1)
    konst = b0 * h1 + (1.0 - b0) * l1
    zeilen = np.repeat(np.arange(folgen), 2)
    spalten = np.arange(nz)
    werte = np.tile([1.0, -1.0], folgen)
    a_eq = sparse.csr_matrix((werte, (zeilen, spalten)), shape=(folgen, nz))
    res = optimize.linprog(cvec, A_eq=a_eq, b_eq=np.zeros(folgen), bounds=[(0, 1)] * nz, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP ohne Lösung: {res.message}")
    return float(-res.fun + konst - float(c) * n), nz, folgen


# ---------------------------------------------------------------------------------------------------------------------
# Verfahren 4: gemischt-ganzzahliges Modell des adaptiven Stoppens
# ---------------------------------------------------------------------------------------------------------------------
def adaptiv_mip(b0, alts, c, tests, N, luecke=MIP_LUECKE):
    """Adaptives Stoppen als gemischt-ganzzahliges Modell: eine Wertvariable je Historienknoten, eine Binärvariable je innerem Knoten (1 = jetzt entscheiden),
    zwei Zeilen je innerem Knoten (Große-M). Rückgabe (Wert, Zahl der Variablen, Zahl der Binärvariablen, Zahl der Zeilen)."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    big = MIP_GROSSM
    nk = 2 ** (N + 1) - 1
    ni = 2 ** N - 1
    nv = nk + ni
    bel = [None] * nk
    bel[0] = b0
    pr = [None] * nk
    for h in range(ni):
        j = (h + 1).bit_length() - 1
        (p1, b1), (p2, b2) = update(bel[h], tests[j], True), update(bel[h], tests[j], False)
        bel[2 * h + 1], bel[2 * h + 2] = b1, b2
        pr[h] = (float(p1), float(p2))
    lb = np.full(nv, -big)
    ub = np.full(nv, big)
    for h in range(ni, nk):
        lb[h] = ub[h] = float(stop_wert(bel[h], alts))
    zi, sj, sv, rhs = [], [], [], []
    zeile = 0
    for h in range(ni):
        s = nk + h
        st = float(stop_wert(bel[h], alts))
        zi += [zeile, zeile]
        sj += [h, s]
        sv += [1.0, big]
        rhs.append(st + big)                     # v_h + M s_h <= Stopp-Wert + M
        zeile += 1
        p1, p2 = pr[h]
        zi += [zeile] * 4
        sj += [h, 2 * h + 1, 2 * h + 2, s]
        sv += [1.0, -p1, -p2, -big]
        rhs.append(-float(c))                    # v_h - p1 v_+ - p2 v_- - M s_h <= -c
        zeile += 1
    if ni == 0:
        return float(stop_wert(b0, alts)), nv, 0, 0
    a_ub = sparse.csr_matrix((sv, (zi, sj)), shape=(zeile, nv))
    cvec = np.zeros(nv)
    cvec[0] = -1.0
    integrality = np.zeros(nv)
    integrality[nk:] = 1
    lb[nk:] = 0.0
    ub[nk:] = 1.0
    opts = {"mip_rel_gap": luecke} if luecke is not None else {}
    res = milp(cvec, constraints=LinearConstraint(a_ub, -np.inf, np.array(rhs)), bounds=Bounds(lb, ub), integrality=integrality, options=opts)
    if res.fun is None:
        raise RuntimeError(f"MIP ohne Lösung: {res.message}")
    return float(-res.fun), nv, ni, zeile


# ---------------------------------------------------------------------------------------------------------------------
# Strategie lesbar machen
# ---------------------------------------------------------------------------------------------------------------------
def strategie_text(strategie, alt_namen=ALT_KURZ, pfad="Start"):
    """Strategie als Zeilen „nach + − : …“. Rückgabe: Liste von (Einrückung, Text) in Baumordnung."""
    zeilen = []

    def rec(s, folge):
        name = "Start" if not folge else "nach " + " ".join(folge)
        if s[0] == "stop":
            zeilen.append((len(folge), f"{name}: stoppen und {alt_namen[s[1]]} wählen"))
        else:
            zeilen.append((len(folge), f"{name}: noch einen Test"))
            rec(s[1][0], folge + ["+"])
            rec(s[1][1], folge + ["−"])

    rec(strategie, [])
    return zeilen


def strategie_kurz(strategie, alt_namen=ALT_KURZ):
    """Einzeiler: „Start: Test; nach +: G; nach −: Test; …“ – nur die erreichbaren Knoten."""
    teile = []

    def rec(s, folge):
        name = "Start" if not folge else "nach " + "".join(folge)
        if s[0] == "stop":
            teile.append(f"{name}: {alt_namen[s[1]]}")
        else:
            teile.append(f"{name}: Test")
            rec(s[1][0], folge + ["+"])
            rec(s[1][1], folge + ["−"])

    rec(strategie, [])
    return "; ".join(teile)


def strategie_tiefe(strategie):
    """Größte Zahl der Tests entlang eines Pfades der Strategie."""
    if strategie[0] == "stop":
        return 0
    return 1 + max(strategie_tiefe(strategie[1][0]), strategie_tiefe(strategie[1][1]))


# ---------------------------------------------------------------------------------------------------------------------
# Kennzahlen des Kapazitätsfalls ohne Test (Nenner immer genannt)
# ---------------------------------------------------------------------------------------------------------------------
def ohne_test(b0=Fraction(1, 2), alts=A_CAP):
    """Wert ohne Test (RP), Hellseher-Wert (WS), EVPI und beide Anteile, in Brüchen."""
    rp = stop_wert(b0, alts)
    ws = b0 * max(a[0] for a in alts) + (1 - b0) * max(a[1] for a in alts)
    evpi = ws - rp
    return {"rp": rp, "ws": ws, "evpi": evpi, "evpi_von_rp": evpi / rp if rp else None, "evpi_von_ws": evpi / ws if ws else None}


def evsi_ein_test(b0, alts, q):
    """Wert der Information eines einzelnen Tests ohne Kosten (EVSI) bei Güte q."""
    return fest_wert(b0, alts, 0, tests_gleich(q, 1), 1) - stop_wert(b0, alts)


# ---------------------------------------------------------------------------------------------------------------------
# Dreistufiger Fall (Kapazität, Nachfrage, Nachbesserung): Baum über alle Knickstellen gegen Szenario-LP
# ---------------------------------------------------------------------------------------------------------------------
D1 = (70, 100, 150)
P1 = (Fraction(3, 10), Fraction(2, 5), Fraction(3, 10))
FAKTOR = (Fraction(4, 5), Fraction(1), Fraction(5, 4))
PFAKTOR = (Fraction(1, 4), Fraction(1, 2), Fraction(1, 4))
PREIS1, PREIS2, ZUKAUF, ZUKAUF_MAX, FEHLMENGE = Fraction(1), Fraction(3, 2), Fraction(19, 10), Fraction(50), Fraction(3)


def _fehlkosten(kapazitaet, d):
    fehl = max(d - kapazitaet, 0)
    zu = min(fehl, ZUKAUF_MAX)
    return ZUKAUF * zu + FEHLMENGE * (fehl - zu)


def _stufe2(x1, d1):
    d2 = [d1 * m for m in FAKTOR]
    kand = sorted({Fraction(x1)} | {max(d, x1) for d in d2} | {max(d - ZUKAUF_MAX, x1) for d in d2})
    return min(PREIS2 * (cx - x1) + sum(PFAKTOR[k] * _fehlkosten(cx, d2[k]) for k in range(3)) for cx in kand)


def dreistufig_wert(x1):
    """Erwartete Kosten bei Kapazität x1 in der ersten Stufe, danach beste Nachbesserung je Nachfrage-Knoten (Rollback in Brüchen)."""
    x1 = Fraction(x1)
    return PREIS1 * x1 + sum(P1[i] * _stufe2(x1, D1[i]) for i in range(3))


def dreistufig_baum():
    """Baum über alle Knickstellen der ersten Stufe, exakt in Brüchen: (bester Wert, x1, Zahl der geprüften Knickstellen)."""
    kand = sorted({Fraction(0)} | {d * m for d in D1 for m in FAKTOR} | {max(d * m - ZUKAUF_MAX, 0) for d in D1 for m in FAKTOR})
    best = min((dreistufig_wert(x), x) for x in kand)
    return best[0], best[1], len(kand)


def dreistufig_lp():
    """Szenario-LP mit nichtantizipativer Nachbesserung (eine je Nachfrage-Knoten der ersten Stufe): (Wert, x1, Variablen, Szenarien)."""
    leaves = [(i, j) for i in range(3) for j in range(3)]
    ns = len(leaves)
    nv = 1 + 3 + 2 * ns
    cc = np.zeros(nv)
    cc[0] = float(PREIS1)
    a = np.zeros((ns, nv))
    b = np.zeros(ns)
    for s, (i, j) in enumerate(leaves):
        pr = float(P1[i] * PFAKTOR[j])
        cc[1 + i] += float(PREIS2) * pr
        cc[4 + s] = float(ZUKAUF) * pr
        cc[4 + ns + s] = float(FEHLMENGE) * pr
        a[s, 0] = a[s, 1 + i] = a[s, 4 + s] = a[s, 4 + ns + s] = -1.0
        b[s] = -float(D1[i] * FAKTOR[j])
    bnds = [(0, None)] * 4 + [(0, float(ZUKAUF_MAX))] * ns + [(0, None)] * ns
    res = optimize.linprog(cc, A_ub=a, b_ub=b, bounds=bnds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return float(res.fun), float(res.x[0]), nv, ns

"""Exponentialnutzen: Sicherheitsäquivalent im Knoten, Tangenten-LP und der Ein-Standort-Fall (Kosten) für Nenner und Kapazitätswahl.

Sicherheitsäquivalent (CARA, Risikotoleranz rho in GE): CE = -rho ln E[exp(-Gewinn/rho)] bei Gewinnen; bei Kosten CE = rho ln E[exp(Kosten/rho)].
Das Szenario-LP kann den Exponentialnutzen exakt nicht darstellen (nichtlinear); stückweise linear, mit m Tangenten an exp((c - cref)/rho), schon näherungsweise.

Ein-Standort-Fall (Zahlen wie in der Nachbar-Demo `newsvendor-evpi-demo`, hier neu gerechnet): Kapazität kostet 1 GE je Einheit, Zukauf 1,9 GE je Einheit bis 50 Einheiten,
darüber Fehlmenge 3 GE je Einheit; Nachfrage lognormal mit Mittel 100 und Log-Streuung 0,35, in n gleich wahrscheinlichen Quantil-Mittelpunkten."""

import math

import numpy as np
from scipy import optimize, sparse, stats

KAP_PREIS, ZUKAUF, ZUKAUF_MAX, FEHLMENGE = 1.0, 1.9, 50.0, 3.0
NACHFRAGE_MITTEL, NACHFRAGE_SIGMA = 100.0, 0.35
TANGENTEN_VON, TANGENTEN_BIS = 60.0, 700.0
X_SUCHE = (50.0, 600.0)


# ---------------------------------------------------------------------------------------------------------------------
# Sicherheitsäquivalent im Knoten (Gewinne)
# ---------------------------------------------------------------------------------------------------------------------
def ce_gewinn(h, n_, b, rho):
    """Sicherheitsäquivalent (GE) einer Alternative mit Gewinn h (hoch, Wahrscheinlichkeit b) und n_ (niedrig); numerisch stabil (Maximum wird abgezogen)."""
    a, c = -h / rho, -n_ / rho
    mx = max(a, c)
    return -rho * (mx + math.log(b * math.exp(a - mx) + (1.0 - b) * math.exp(c - mx)))


def ce_verteilung(gewinne, rho):
    """Sicherheitsäquivalent einer Verteilung gleich wahrscheinlicher Gewinne."""
    g = np.asarray(gewinne, dtype=float)
    a = -g / rho
    mx = a.max()
    return float(-rho * (mx + math.log(np.mean(np.exp(a - mx)))))


def wahl_ew(alts, b):
    return 1 if (b * alts[1][0] + (1 - b) * alts[1][1]) > (b * alts[0][0] + (1 - b) * alts[0][1]) else 0


def wahl_ce(alts, b, rho):
    return 1 if ce_gewinn(alts[1][0], alts[1][1], b, rho) > ce_gewinn(alts[0][0], alts[0][1], b, rho) else 0


def wechsel_zaehlung(probleme, rhos, toleranz=1e-9):
    """Anteil der nicht dominierten Probleme, in denen die Wahl nach Erwartungswert (risikoneutrales LP) eine andere Alternative ergibt als die Wahl nach dem Sicherheitsäquivalent.
    Probleme, in denen eine der beiden Wahlen auf einem Gleichstand (Abstand unter `toleranz`) beruht, zählen nicht (die Wahl dort hängt an der letzten Stelle).
    Rückgabe: Zahl der gezählten Probleme und je rho (Zahl der Wechsel, mittlerer Verlust in Sicherheitsäquivalent-GE, größter Verlust)."""
    from evw_scenario import dominiert

    gezaehlt = 0
    wechsel = {r: [] for r in rhos}
    for p in probleme:
        if dominiert(p.alts):
            continue
        ew = [p.b0 * h + (1 - p.b0) * n_ for h, n_ in p.alts]
        if abs(ew[0] - ew[1]) < toleranz:
            continue
        gezaehlt += 1
        ia = 0 if ew[0] > ew[1] else 1
        for r in rhos:
            ce = [ce_gewinn(h, n_, p.b0, r) for h, n_ in p.alts]
            if abs(ce[0] - ce[1]) < toleranz:
                continue
            ib = 0 if ce[0] > ce[1] else 1
            if ia != ib:
                wechsel[r].append(ce[ib] - ce[ia])
    ergebnis = {}
    for r in rhos:
        v = wechsel[r]
        ergebnis[r] = {"wechsel": len(v), "verlust_mittel": float(np.mean(v)) if v else 0.0, "verlust_max": float(max(v)) if v else 0.0}
    return gezaehlt, ergebnis


# ---------------------------------------------------------------------------------------------------------------------
# Ein-Standort-Fall (Kosten)
# ---------------------------------------------------------------------------------------------------------------------
def nachfrage(n):
    """n gleich wahrscheinliche Quantil-Mittelpunkte einer Log-Normalverteilung mit Mittel 100 und Log-Streuung 0,35."""
    return stats.lognorm.ppf((np.arange(n) + 0.5) / n, s=NACHFRAGE_SIGMA, scale=NACHFRAGE_MITTEL * math.exp(-0.5 * NACHFRAGE_SIGMA ** 2))


def nachbesserung_kosten(x, d):
    """Kosten (ohne Kapazität) bei Kapazität x und Nachfrage d: Zukauf bis 50 Einheiten zu 1,9, der Rest als Fehlmenge zu 3."""
    fehl = np.maximum(d - x, 0.0)
    zu = np.minimum(fehl, ZUKAUF_MAX)
    return ZUKAUF * zu + FEHLMENGE * (fehl - zu)


def kosten(x, d):
    return KAP_PREIS * x + nachbesserung_kosten(x, d)


def ce_kosten(werte, rho):
    """Sicherheitsäquivalent der Kosten: rho ln E exp(Kosten/rho), stabil."""
    w = np.asarray(werte, dtype=float)
    mx = w.max()
    return float(rho * (math.log(np.mean(np.exp((w - mx) / rho))) + mx / rho))


def beste_kapazitaet_ce(d, rho):
    """Kapazität mit kleinstem Sicherheitsäquivalent (skalare Suche): (x, CE)."""
    res = optimize.minimize_scalar(lambda x: ce_kosten(kosten(x, d), rho), bounds=X_SUCHE, method="bounded", options={"xatol": 1e-9})
    return float(res.x), float(res.fun)


def beste_kapazitaet_ew(d):
    """Baum (Rollback über die Szenarien): beste Kapazität nach dem Erwartungswert der Kosten, aus allen Knickstellen {d_s, d_s - 50} geprüft: (x, RP)."""
    kand = np.unique(np.concatenate([d, d - ZUKAUF_MAX, [0.0]]))
    kand = kand[kand >= 0.0]
    werte = np.array([np.mean(kosten(x, d)) for x in kand])
    i = int(np.argmin(werte))
    return float(kand[i]), float(werte[i])


def szenario_lp_ew(d):
    """Szenario-LP des Erwartungswerts (eine Kapazität für alle Szenarien, Zukauf und Fehlmenge je Szenario): (RP, x)."""
    n = len(d)
    nv = 1 + 2 * n
    cvec = np.concatenate([[KAP_PREIS], np.full(n, ZUKAUF / n), np.full(n, FEHLMENGE / n)])
    zeilen = np.repeat(np.arange(n), 3)
    spalten = np.stack([np.zeros(n, dtype=int), 1 + np.arange(n), 1 + n + np.arange(n)], axis=1).ravel()
    a = sparse.csr_matrix((np.full(3 * n, -1.0), (zeilen, spalten)), shape=(n, nv))
    bounds = [(0, None)] + [(0, ZUKAUF_MAX)] * n + [(0, None)] * n
    res = optimize.linprog(cvec, A_ub=a, b_ub=-d, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return float(res.fun), float(res.x[0])


def _call_lognormal(a):
    """E[max(D - a, 0)] für die Log-Normalverteilung der Nachfrage (geschlossene Form, Black-Scholes-Gestalt)."""
    m, s = NACHFRAGE_MITTEL, NACHFRAGE_SIGMA
    if a <= 0:
        return m - a
    d2 = (math.log(m / a) - 0.5 * s * s) / s
    return m * stats.norm.cdf(d2 + s) - a * stats.norm.cdf(d2)


def ein_standort_geschlossen():
    """Ein-Standort-Fall in geschlossener Form (stetige Nachfrage): RP, EEV, WS, x_RP, EVPI, VSS in Kosteneinheiten (GE)."""
    def erwartet(x):
        return (KAP_PREIS * x + ZUKAUF * (_call_lognormal(x) - _call_lognormal(x + ZUKAUF_MAX)) + FEHLMENGE * _call_lognormal(x + ZUKAUF_MAX))

    res = optimize.minimize_scalar(erwartet, bounds=(0.0, NACHFRAGE_MITTEL * math.exp(4 * NACHFRAGE_SIGMA)), method="bounded", options={"xatol": 1e-10})
    rp, x = float(res.fun), float(res.x)
    eev = float(erwartet(NACHFRAGE_MITTEL))
    ws = KAP_PREIS * NACHFRAGE_MITTEL
    return {"x_rp": x, "rp": rp, "eev": eev, "ws": ws, "evpi": rp - ws, "vss": eev - rp}


def nenner_tabelle(evpi, vss, rp, ws, eev):
    """EVPI und VSS in Prozent der drei Nenner RP, WS und EEV (Kosten)."""
    return {"RP": (100 * evpi / rp, 100 * vss / rp), "WS": (100 * evpi / ws, 100 * vss / ws), "EEV": (100 * evpi / eev, 100 * vss / eev)}


# ---------------------------------------------------------------------------------------------------------------------
# Tangenten-LP
# ---------------------------------------------------------------------------------------------------------------------
def tangenten(m, rho, cref, von=TANGENTEN_VON, bis=TANGENTEN_BIS):
    """m Tangenten an u(c) = exp((c - cref)/rho), gleichmäßig in [von, bis]: (Stützstellen, Steigungen a_j, Achsenabschnitte b_j)."""
    stuetz = np.linspace(von, bis, m) if m > 1 else np.array([(von + bis) / 2.0])
    u = np.exp((stuetz - cref) / rho)
    a = u / rho
    return stuetz, a, u - a * stuetz


def tangenten_lp(d, rho, m, cref, von=TANGENTEN_VON, bis=TANGENTEN_BIS):
    """Szenario-LP mit stückweise linear (m Tangenten) angenähertem Exponentialnutzen der Kosten: minimiere die Summe p_s t_s mit t_s >= a_j Kosten_s + b_j.
    Variablen: x, Zukauf je Szenario, Fehlmenge je Szenario, t je Szenario. Rückgabe (Kapazität x, Zahl der Variablen, Zahl der Ungleichungen)."""
    n = len(d)
    nv = 1 + 3 * n
    _, a, b = tangenten(m, rho, cref, von, bis)
    cvec = np.zeros(nv)
    cvec[1 + 2 * n:] = 1.0 / n
    # Bedarf: -x - Zukauf - Fehlmenge <= -d
    z1 = np.repeat(np.arange(n), 3)
    s1 = np.stack([np.zeros(n, dtype=int), 1 + np.arange(n), 1 + n + np.arange(n)], axis=1).ravel()
    bedarf = sparse.csr_matrix((np.full(3 * n, -1.0), (z1, s1)), shape=(n, nv))
    # Tangente j, Szenario s: a_j (x + ZUKAUF zu_s + FEHLMENGE fm_s) - t_s <= -b_j
    zeile = np.arange(m * n)
    js = np.repeat(np.arange(m), n)
    ss = np.tile(np.arange(n), m)
    rows = np.repeat(zeile, 4)
    cols = np.stack([np.zeros(m * n, dtype=int), 1 + ss, 1 + n + ss, 1 + 2 * n + ss], axis=1).ravel()
    vals = np.stack([a[js] * KAP_PREIS, a[js] * ZUKAUF, a[js] * FEHLMENGE, np.full(m * n, -1.0)], axis=1).ravel()
    tang = sparse.csr_matrix((vals, (rows, cols)), shape=(m * n, nv))
    a_ub = sparse.vstack([bedarf, tang], format="csr")
    b_ub = np.concatenate([-d, -b[js]])
    bounds = [(0, None)] + [(0, ZUKAUF_MAX)] * n + [(0, None)] * n + [(None, None)] * n
    res = optimize.linprog(cvec, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return float(res.x[0]), nv, n + m * n


def mehrverlust(d, rho, x_lp, ce_exakt):
    """Mehrverlust in GE (Sicherheitsäquivalent, exakt bewertet) der Kapazität aus dem Tangenten-LP gegen die exakt beste Kapazität."""
    return ce_kosten(kosten(x_lp, d), rho) - ce_exakt

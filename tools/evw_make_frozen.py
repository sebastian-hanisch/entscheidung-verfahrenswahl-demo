"""Erzeugt `evw_frozen.json`: die vorgerechneten Messreihen der Demo (zu schwer für die Live-Rechnung der App).

    python tools/evw_make_frozen.py            # alles rechnen und evw_frozen.json schreiben (einige Minuten: viele LPs)
    python tools/evw_make_frozen.py --check    # schnelle Teile neu rechnen und mit der Datei vergleichen (Zählungen als Band), kein Schreiben

Vorgerechnet sind: (1) vier Zufalls-Seeds mit je 1000 Zufallsproblemen (Baum, Zustandsrechnung, Szenario-LP je feste Testzahl und gemischt-ganzzahliges Modell),
(2) der Entscheidungswechsel bei Exponentialnutzen (20 000 Probleme), (3) die Tangenten-LPs des Ein-Standort-Falls mit 2000 Szenarien.
Der Zufall kommt nur aus dem SplitMix64 der Demo (`evw_rng.py`). Plattform: die Zählungen sind Zählungen über Gleitkomma-Vergleiche und gelten als Band (± 2 Probleme);
LP- und MIP-Werte gelten auf 1e-7 bzw. 1e-5, nie bitgleich."""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evw_exponential as E  # noqa: E402
import evw_kern as K  # noqa: E402
import evw_scenario as S  # noqa: E402

DATEI = ROOT / "evw_frozen.json"
SEED_HAUPT = 20261120
SEEDS_GEGENPROBE = (1, 777, 424242)
SEED_WECHSEL = 20261121
M_PROBLEME, N_TESTS, MIP_ANZAHL = 1000, 4, 300
RHOS_WECHSEL = (20.0, 50.0, 100.0, 300.0)
RHOS_TANGENTEN = (500.0, 200.0, 100.0, 50.0, 25.0)
MS_TANGENTEN = (4, 8, 16, 32, 64)
N_SZENARIEN = 2000


def zufall_seed(seed, m=M_PROBLEME, n_tests=N_TESTS, lp=True, mip=True, mip_anzahl=MIP_ANZAHL):
    """Zählungen über m Zufallsprobleme mit bis zu n_tests gleichen Tests nacheinander."""
    probleme = S.ziehe_probleme(seed, m)
    z = dict(m=m, N=n_tests, nicht_dominiert=0, test_lohnt=0, adaptiv_besser=0, dp_abweichungen=0)
    luecken, luecken_lohnt = [], []
    lp_geprueft = lp_abw = mip_geprueft = mip_abw = 0
    lp_max = mip_max = 0.0
    for k, p in enumerate(probleme):
        if not S.dominiert(p.alts):
            z["nicht_dominiert"] += 1
        tests = K.tests_gleich(p.q, n_tests)
        v_baum, _ = K.baum(p.b0, p.alts, p.c, tests)
        v_dp = K.dp_wert_float(p.b0, p.alts, p.c, p.q, n_tests)
        if abs(v_baum - v_dp) > 1e-9:
            z["dp_abweichungen"] += 1
        fest = [K.fest_wert(p.b0, p.alts, p.c, tests, n) for n in range(n_tests + 1)]
        luecke = v_baum - max(fest)
        luecken.append(luecke)
        lohnt = v_baum > K.stop_wert(p.b0, p.alts) + 1e-9
        if lohnt:
            z["test_lohnt"] += 1
            luecken_lohnt.append(luecke)
        if luecke > 1e-7:
            z["adaptiv_besser"] += 1
        if lp:
            for n in range(n_tests + 1):
                w = K.fest_lp(p.b0, p.alts, p.c, tests, n)[0]
                lp_geprueft += 1
                lp_max = max(lp_max, abs(w - fest[n]))
                if abs(w - fest[n]) > 1e-7:
                    lp_abw += 1
        if mip and k < mip_anzahl:
            w = K.adaptiv_mip(p.b0, p.alts, p.c, tests, n_tests)[0]
            mip_geprueft += 1
            mip_max = max(mip_max, abs(w - v_baum))
            if abs(w - v_baum) > 1e-5:
                mip_abw += 1
    a = np.array(luecken_lohnt) if luecken_lohnt else np.array([0.0])
    z.update(
        adaptiv_besser_unter_lohnend=sum(1 for x in luecken_lohnt if x > 1e-7),
        luecke_mittel_lohnend=float(a.mean()), luecke_median_lohnend=float(np.median(a)), luecke_max=float(a.max()), luecke_mittel_alle=float(np.mean(luecken)),
        luecke_min=float(min(luecken)),
    )
    if lp:
        z.update(lp_geprueft=lp_geprueft, lp_abweichungen=lp_abw, lp_max_abweichung=lp_max)
    if mip:
        z.update(mip_geprueft=mip_geprueft, mip_abweichungen=mip_abw, mip_max_abweichung=mip_max)
    return z


def mip_standardluecke(seed=777, anzahl=MIP_ANZAHL, n_tests=N_TESTS):
    """Dieselben Probleme mit der Standardlücke von HiGHS (relative Lücke 1e-4): wie viele Werte liegen um mehr als 1e-5 unter dem Baumwert."""
    probleme = S.ziehe_probleme(seed, anzahl)
    abw, groesste = 0, 0.0
    for p in probleme:
        tests = K.tests_gleich(p.q, n_tests)
        v_baum, _ = K.baum(p.b0, p.alts, p.c, tests)
        w = K.adaptiv_mip(p.b0, p.alts, p.c, tests, n_tests, luecke=None)[0]
        d = v_baum - w
        if abs(d) > 1e-5:
            abw += 1
        groesste = max(groesste, abs(d))
    return {"seed": seed, "geprueft": anzahl, "abweichungen": abw, "groesste_abweichung": groesste}


def wechsel(seed=SEED_WECHSEL, m=20000, rhos=RHOS_WECHSEL):
    gezaehlt, erg = E.wechsel_zaehlung(S.ziehe_probleme(seed, m), list(rhos))
    return {"seed": seed, "m": m, "nicht_dominiert": gezaehlt, "rho": {str(int(r)): v for r, v in erg.items()}}


def wechsel_gegenprobe(seeds=SEEDS_GEGENPROBE, m=20000, rho=50.0):
    aus = {}
    for s in seeds:
        gezaehlt, erg = E.wechsel_zaehlung(S.ziehe_probleme(s, m), [rho])
        aus[str(s)] = {"nicht_dominiert": gezaehlt, "wechsel": erg[rho]["wechsel"], "anteil": erg[rho]["wechsel"] / gezaehlt}
    return aus


def tangenten_zeilen(rhos=RHOS_TANGENTEN, ms=MS_TANGENTEN, n=N_SZENARIEN):
    d = E.nachfrage(n)
    zeilen = []
    for rho in rhos:
        x_ex, ce_ex = E.beste_kapazitaet_ce(d, rho)
        for m in ms:
            t0 = time.time()
            x_lp, _, _ = E.tangenten_lp(d, rho, m, ce_ex)
            sek = time.time() - t0
            zeilen.append({"rho": rho, "m": m, "x_exakt": x_ex, "ce_exakt": ce_ex, "x_lp": x_lp, "mehrverlust": E.mehrverlust(d, rho, x_lp, ce_ex), "sekunden": round(sek, 2)})
    return zeilen


def alles():
    t0 = time.time()
    daten = {"version": 1, "plattform": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "system": platform.system()}}
    daten["zufall"] = {}
    for seed in (SEED_HAUPT,) + SEEDS_GEGENPROBE:
        print(f"Zufalls-Seed {seed} ...", flush=True)
        daten["zufall"][str(seed)] = zufall_seed(seed)
    print("MIP mit Standardlücke ...", flush=True)
    daten["mip_standardluecke"] = mip_standardluecke()
    print("Entscheidungswechsel ...", flush=True)
    daten["wechsel"] = wechsel()
    daten["wechsel_gegenprobe_rho50"] = wechsel_gegenprobe()
    print("Tangenten-LPs (n = 2000) ...", flush=True)
    daten["tangenten"] = {"n": N_SZENARIEN, "von": E.TANGENTEN_VON, "bis": E.TANGENTEN_BIS, "zeilen": tangenten_zeilen()}
    daten["laufzeit_sekunden"] = round(time.time() - t0)
    return daten


def pruefe(daten):
    """Schnelle Teile neu rechnen und mit `daten` vergleichen. Rückgabe: Liste der Abweichungen (leer = in Ordnung)."""
    fehler = []

    def nah(name, a, b, tol):
        if abs(a - b) > tol:
            fehler.append(f"{name}: neu {a}, eingefroren {b}")

    for seed, z in daten["zufall"].items():
        neu = zufall_seed(int(seed), lp=False, mip=False)
        for key in ("nicht_dominiert", "test_lohnt", "adaptiv_besser", "adaptiv_besser_unter_lohnend"):
            nah(f"Zufalls-Seed {seed} {key}", neu[key], z[key], 2)
        nah(f"Zufalls-Seed {seed} Lücke Mittel", neu["luecke_mittel_lohnend"], z["luecke_mittel_lohnend"], 1e-6)
        if z["dp_abweichungen"] != 0 or neu["dp_abweichungen"] != 0:
            fehler.append(f"Zufalls-Seed {seed}: Baum und Zustandsrechnung weichen ab")
    if daten["zufall"][str(SEED_HAUPT)].get("lp_abweichungen", 1) != 0 or daten["zufall"][str(SEED_HAUPT)].get("mip_abweichungen", 1) != 0:
        fehler.append("eingefroren: LP oder MIP weichen vom Baum ab")
    w = daten["wechsel"]
    neu = wechsel(w["seed"], w["m"])
    nah("Entscheidungswechsel nicht dominiert", neu["nicht_dominiert"], w["nicht_dominiert"], 2)
    for r, v in w["rho"].items():
        nah(f"Entscheidungswechsel rho = {r}", neu["rho"][r]["wechsel"], v["wechsel"], 3)
    d = E.nachfrage(daten["tangenten"]["n"])
    for z in daten["tangenten"]["zeilen"]:
        if z["m"] <= 8:
            ce_ex = E.beste_kapazitaet_ce(d, z["rho"])[1]
            x_lp = E.tangenten_lp(d, z["rho"], z["m"], ce_ex)[0]
            nah(f"Tangenten rho = {z['rho']}, m = {z['m']}: Mehrverlust", E.mehrverlust(d, z["rho"], x_lp, ce_ex), z["mehrverlust"], 0.02 + 0.05 * abs(z["mehrverlust"]))
    return fehler


def laden():
    with open(DATEI, encoding="utf-8") as f:
        return json.load(f)


def main(argv):
    if "--check" in argv:
        fehler = pruefe(laden())
        for f in fehler:
            print("ABWEICHUNG", f)
        print("eingefroren stimmt mit der Neu-Rechnung überein" if not fehler else f"{len(fehler)} Abweichungen")
        return 1 if fehler else 0
    daten = alles()
    with open(DATEI, "w", encoding="utf-8", newline="\n") as f:
        json.dump(daten, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"geschrieben: {DATEI} ({daten['laufzeit_sekunden']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

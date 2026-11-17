"""Zufallsgenerator: Kopie der geprüften Vorlage (Hash), bekannte Folge, Bereiche. Kein Python-`hash()`, kein `random`, kein `numpy.random` in der Rechnung."""

import ast
import hashlib
from pathlib import Path

import evw_rng as R

ROOT = Path(__file__).resolve().parent.parent
VORLAGE_SHA256 = "d69b2b2fd67c1234cc0d79249676a662a2189d02803eea7d2edfb04b7a505a3f"   # tools/templates/rng_splitmix64_exact.py im Website-Repo


def test_evw_rng_ist_die_unveraenderte_vorlage():
    assert hashlib.sha256((ROOT / "evw_rng.py").read_bytes()).hexdigest() == VORLAGE_SHA256


def test_bekannte_splitmix_folge_fuer_zustand_null():
    """Referenzfolge von SplitMix64 mit Zustand 0 (Vigna): 0xe220a8397b1dcdaf, 0x6e789e6aa1b965f4."""
    g = R.SplitMix64(0)
    assert g.next_u64() == 0xE220A8397B1DCDAF and g.next_u64() == 0x6E789E6AA1B965F4


def test_random_und_below_liegen_im_bereich_und_sind_reproduzierbar():
    a, b = R.SplitMix64(20261120), R.SplitMix64(20261120)
    ua = [a.random() for _ in range(200)]
    assert ua == [b.random() for _ in range(200)] and all(0.0 <= u < 1.0 for u in ua)
    g = R.SplitMix64(7)
    werte = [g.below(301) for _ in range(2000)]
    assert min(werte) == 0 and max(werte) == 300


def test_die_rechnung_benutzt_keinen_anderen_zufall():
    verboten = {"random", "secrets", "hashlib"}
    for datei in ROOT.glob("evw_*.py"):
        if datei.name in ("evw_rng.py", "evw_presets.py"):      # der Knopf „Neuen Zufalls-Seed ziehen“ ruft random.randint auf (kein Teil einer Messung)
            continue
        baum = ast.parse(datei.read_text(encoding="utf-8"))
        for knoten in ast.walk(baum):
            namen = [a.name.split(".")[0] for a in knoten.names] if isinstance(knoten, ast.Import) else ([knoten.module.split(".")[0]] if isinstance(knoten, ast.ImportFrom) and knoten.module else [])
            assert not set(namen) & verboten, (datei.name, namen)
            if isinstance(knoten, ast.Call) and getattr(knoten.func, "id", "") == "hash":
                raise AssertionError(f"{datei.name}: Python-hash() ist je Prozess gesalzen und kein Zufalls-Seed")
        assert "numpy.random" not in datei.read_text(encoding="utf-8").replace("`numpy.random`", ""), datei.name

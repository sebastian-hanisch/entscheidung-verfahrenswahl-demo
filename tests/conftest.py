"""Gemeinsame Einrichtung: der Wurzelordner steht im Importpfad; die Mini-Instanz ist das Handbeispiel (DEMO-PLAYBOOK §1b)."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evw_kern as K  # noqa: E402


@pytest.fixture
def hand():
    """Handbeispiel: Kapazitätsfall (G: +200 / −100, K: +30, P(hoch) = 1/2), Tests mit q = 4/5, 5 GE je Test.
    Handrechnung: ohne Test 50; ein Test 85 − 5 = 80; zwei Tests nacheinander mit Stoppen 807/10 = 80,7 (nach + stoppen, nach − noch einmal testen)."""
    return {"b0": Fraction(1, 2), "alts": K.A_CAP, "q": Fraction(4, 5), "c": Fraction(5)}

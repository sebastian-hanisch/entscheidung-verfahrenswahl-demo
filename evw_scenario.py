"""Zufallsprobleme: zwei Alternativen, zwei Zustände, ein Test (ausgedacht, kein Fall).

Ein Problem hat je Alternative eine Auszahlung bei hoher und bei niedriger Nachfrage (ganzzahlig gleichverteilt in [-100, 200] GE), die Wahrscheinlichkeit
P(hoch) (gleichverteilt in [0,1; 0,9]), die Testgüte q (gleichverteilt in [0,55; 0,95]) und die Kosten c je Test (gleichverteilt in [0; 10] GE).
Zufall nur aus dem ganzzahligen SplitMix64 (`evw_rng.py`); die Reihenfolge der Ziehungen ist festgelegt und gehört zur Definition."""

from dataclasses import dataclass

from evw_rng import SplitMix64

AUSZAHLUNG_MIN, AUSZAHLUNG_MAX = -100, 200
B_MIN, B_MAX = 0.1, 0.9
Q_MIN, Q_MAX = 0.55, 0.95
C_MIN, C_MAX = 0.0, 10.0


@dataclass(frozen=True)
class Problem:
    alts: tuple       # ((G_hoch, G_niedrig), (K_hoch, K_niedrig))
    b0: float
    q: float
    c: float


def ziehe_problem(rng: SplitMix64) -> Problem:
    """Ein Problem; Reihenfolge der Ziehungen: vier Auszahlungen (G hoch, G niedrig, K hoch, K niedrig), P(hoch), q, c."""
    spanne = AUSZAHLUNG_MAX - AUSZAHLUNG_MIN + 1
    g_h = AUSZAHLUNG_MIN + int(rng.below(spanne))
    g_l = AUSZAHLUNG_MIN + int(rng.below(spanne))
    k_h = AUSZAHLUNG_MIN + int(rng.below(spanne))
    k_l = AUSZAHLUNG_MIN + int(rng.below(spanne))
    b0 = B_MIN + (B_MAX - B_MIN) * rng.random()
    q = Q_MIN + (Q_MAX - Q_MIN) * rng.random()
    c = C_MIN + (C_MAX - C_MIN) * rng.random()
    return Problem(((g_h, g_l), (k_h, k_l)), b0, q, c)


def ziehe_probleme(seed: int, m: int) -> list:
    """m Probleme aus dem Zufalls-Seed `seed` (jedes Problem setzt die Folge fort)."""
    rng = SplitMix64(seed)
    return [ziehe_problem(rng) for _ in range(m)]


def dominiert(alts) -> bool:
    """Eine Alternative ist in beiden Zuständen mindestens so gut wie die andere: dann gibt es nichts zu entscheiden und ein Test ist wertlos."""
    (h0, l0), (h1, l1) = alts
    return (h0 >= h1 and l0 >= l1) or (h1 >= h0 and l1 >= l0)

# Verfahrenswahl: Baum, Szenario-LP oder Zustandsrechnung – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-entscheidung-verfahrenswahl-demo.streamlit.app/)** -->

Die letzte von elf Demos zur **Entscheidungsanalyse** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“; die anderen zehn stehen unten bei den Nachbarn und rechnen alle mit **einem** Verfahren, dem Entscheidungsbaum (Ereignisknoten) mit Rollback. Diese Demo hebt die Annahme auf, es sei gleich, mit welchem Verfahren man rechnet: **Entscheidungsbaum, Zustandsrechnung, Szenario-LP und gemischt-ganzzahliges Modell** auf derselben Entscheidung, mit Wert, Größe und Abstand nebeneinander. Vehikel ist ein **Kapazitätsfall mit ausgedachten Zahlen** (Geldeinheiten GE; die Auszahlungen sind erfunden, das Beispiel ist ein Modell, kein Fall).
„Entscheidungsbaum“ meint hier das Rechenverfahren mit Ereignisknoten; der Baum des Verfahrens CART (maschinelles Lernen, [`cart-demo`](https://github.com/sebastian-hanisch/cart-demo)) ist etwas anderes. Die Entscheidungshilfe der Demo heißt **Wegweiser**.
Nachbarn (die anderen Demos zur Entscheidungsanalyse): [`baum-strategie-demo`](https://github.com/sebastian-hanisch/baum-strategie-demo) (Entscheiden mit Baum: Rollback und Strategien), [`testwert-evsi-demo`](https://github.com/sebastian-hanisch/testwert-evsi-demo) (ein Test und sein Wert), [`kipp-schwelle-demo`](https://github.com/sebastian-hanisch/kipp-schwelle-demo) (kippt die Entscheidung?), [`risiko-aequivalent-demo`](https://github.com/sebastian-hanisch/risiko-aequivalent-demo) (Risikoeinstellung), [`information-risikoaversion-demo`](https://github.com/sebastian-hanisch/information-risikoaversion-demo) (Information unter Risikoaversion), [`nutzen-erheben-demo`](https://github.com/sebastian-hanisch/nutzen-erheben-demo) (Nutzen erheben), [`erwartungsnutzen-grenze-demo`](https://github.com/sebastian-hanisch/erwartungsnutzen-grenze-demo) (wo der Erwartungsnutzen endet), [`baum-waechst-demo`](https://github.com/sebastian-hanisch/baum-waechst-demo) (der Baum wächst), [`stichprobenbaum-demo`](https://github.com/sebastian-hanisch/stichprobenbaum-demo) (Szenarien aus einer Stichprobe) und [`attribute-risiko-demo`](https://github.com/sebastian-hanisch/attribute-risiko-demo) (mehrere Attribute unter Risiko).
Weitere Nachbarn: [`newsvendor-evpi-demo`](https://github.com/sebastian-hanisch/newsvendor-evpi-demo) (EVPI und VSS als Rechnung, ein Szenario-LP je Informationsstand; hier nur der Ein-Standort-Fall für die Nenner und das Tangenten-LP, neu gerechnet), [`bestell-rueckwaerts-demo`](https://github.com/sebastian-hanisch/bestell-rueckwaerts-demo) (Rückwärtsrechnung und Zustandszusammenlegung am Lager, Zustand ist die Bestandsposition) und [`entscheiden-lernen-wahl-demo`](https://github.com/sebastian-hanisch/entscheiden-lernen-wahl-demo): eine **andere** Verfahrenswahl, dort für das Entscheiden aus Daten; diese Demo vergleicht Rechenverfahren für ein gegebenes Entscheidungsmodell. Kein CVaR ([`cvar-demo`](https://github.com/sebastian-hanisch/cvar-demo)), kein Stichprobenfehler ([`saa-demo`](https://github.com/sebastian-hanisch/saa-demo)).

## Warum dieses Problem

Baum und Szenario-LP geben im Erwartungswert denselben Wert: wer das einmal gesehen hat, schließt leicht „also egal“. Dieser Schluss gilt bei **fester Informationsstruktur und Erwartungswert**; die Demo zeigt zwei Fälle, in denen er endet (Stoppen nach Beobachtung, Exponentialnutzen). Sie zeigt, **wo das stimmt, wo es aufhört und was es kostet**.
Im Kapazitätsfall geben Baum, Zustandsrechnung, Szenario-LP und gemischt-ganzzahliges Modell denselben Wert (Brüche gleich Brüche: 807/10 bei höchstens zwei Tests nacheinander, dreistufig 2117/16, das LP gleich dem Baum auf 1e-9). Darf aber nach jedem Test gestoppt werden, ist das beste Szenario-LP mit fester Testzahl in rund 20 % der Zufallsprobleme schlechter als der Baum (188 bis 220 von 1000 über vier Zufalls-Seeds; im Hauptlauf mit 220 im Mittel 2,17 GE unter den 252, in denen sich ein Test lohnt, in den anderen Zufalls-Seeds rund 2 GE). Wählt man nach dem Sicherheitsäquivalent statt nach dem Erwartungswert, wählt das risikoneutrale LP in 22,9 % der nicht dominierten Zufallsprobleme bei ρ = 50 eine andere Alternative. Und die Zustandsrechnung spart nur, wo Zustände zusammenfallen: bei N = 14 gleichen Tests 120 Zustände statt 32 767 Knoten, bei N = 14 und Quotenverhältnissen, die multiplikativ unabhängig sind (die ersten 14 Primzahlen), alle 32 767; bei bloß verschiedenen Verhältnissen können Folgen zusammenfallen.
Alle Zahlen gelten für **diesen Kapazitätsfall und diese Zufallsprobleme** (zwei Alternativen, zwei Zustände); Zähleinheit sind Probleme, Knoten und Zustände wie genannt. Alle Formeln und Befunde sind **eigene Nachrechnung in diesem Modell, bei diesen Parametern**. Eine Neuheit wird nicht behauptet: Rollback, Zustandsrechnung und Szenario-LP sind Lehrbuchstoff; neu ist die Vorführbarkeit mit Zähleinheiten und der Vergleich der Größen (siehe Quellen).

## Modell

**Der Fall (ausgedachte Zahlen).** Alternative G (groß bauen) zahlt +200 GE bei hoher und −100 GE bei niedriger Nachfrage, Alternative K (klein bauen oder mieten) sicher +30 GE; P(hoch) = 1/2. Ohne Test: EW(G) = 50, EW(K) = 30, die Wahl ist G. Mit vollkommener Information (Hellseher) wäre der Wert 115 GE; EVPI = 65 GE.
**Test.** Sensitivität = Spezifität = q (Raster 1/20 in der App), Kosten c GE je Test, bis zu N Tests nacheinander; nach jedem Ergebnis darf gestoppt werden („Stoppen nach Beobachtung“) oder die Zahl ist vorab festgelegt („feste Testzahl“). Der **Glaubenszustand** b = P(hoch) wird nach Bayes fortgeschrieben (Formeln im 📐-Expander der App); nach „+“ gilt b′ = b·q / (b·q + (1 − b)(1 − q)).
**Zählweise.** Ein **Historienknoten** je Folge von Testergebnissen der Länge höchstens N: 2^(N+1) − 1 Knoten, davon 2^N − 1 innere (hier folgt ein Test). Ein **Zustand** der Zustandsrechnung ist ein Paar (Tiefe, Glaubenszustand), die verschiedenen Glaubenszustände werden als Brüche exakt gezählt: zwei Folgen mit demselben Bruch sind ein Zustand.
**Strategie** heißt: nach dieser Ergebnisfolge jetzt entscheiden (welche Alternative) oder noch einen Test; bei Gleichstand wird gestoppt (festgelegt, sonst sind Strategien nicht vergleichbar).
**Nenner.** Wo die Demo EVPI (oder VSS) nennt, steht der **Nenner** daneben: RP (Wert ohne Information), WS (Wert mit vollkommener Information). Im Kapazitätsfall (Gewinne) ist EVPI = 65 GE = 56,5 % von WS (13/23). Als Anteil an RP wäre EVPI im Gewinnfall 130 % (irreführend); die Demo zeigt GE und den Anteil an EVPI und nennt den Nenner. Die 130 % sind 65/50 (RP = 50 GE; Messung B1, `test_b1_ohne_test_evpi_nenner`, und `evpi_von_rp` in `evw_kern.py`). Im Ein-Standort-Fall (Kosten) sind es 22,55 % von RP und 29,11 % von WS.
**Gilt für** den Kapazitätsfall mit zwei Alternativen und zwei Zuständen, den Erwartungswert oder den Exponentialnutzen (Experiment 4), diskrete Verteilungen.

## Methodik

| Verfahren | Wie es arbeitet | Grenze |
|---|---|---|
| **Entscheidungsbaum (Ereignisknoten)** (`evw_kern.baum`) | Rollback von den Blättern her, ohne Gedächtnis: jeder Historienknoten wird einzeln gerechnet; Baum und Zustandsrechnung in Brüchen (`fractions`) | 2^(N+1) − 1 Knoten; ausgeschrieben nur bis N = 3 als Diagramm |
| **Zustandsrechnung** (`evw_kern.dp`) | dieselbe Rückwärtsrechnung mit Gedächtnis auf dem Paar (Tiefe, Glaubenszustand) | spart nur, wo Ergebnisfolgen auf denselben Bruch führen |
| **Szenario-LP** (`evw_kern.fest_lp`) | `scipy.optimize.linprog` (HiGHS): je Szenario (Zustand, Ergebnisfolge) ein Anteil x von G, Nichtantizipativität als Gleichung je Ergebnisfolge; **ein Lauf je feste Testzahl** | kann das Stoppen nach Beobachtung nicht ausdrücken (Experiment 2) |
| **Gemischt-ganzzahliges Modell** (`evw_kern.adaptiv_mip`) | `scipy.optimize.milp`: ein Wert je Historienknoten, eine Binärvariable „jetzt entscheiden“ je innerem Knoten, zwei Große-M-Zeilen je innerem Knoten, `mip_rel_gap` = 1e-9 | in der App bis N = 8 gerechnet (Laufzeit); an Zufallsproblemen nur bei N = 4 geprüft |
| **Tangenten-LP** (`evw_exponential.tangenten_lp`) | Exponentialnutzen stückweise linear: m Tangenten, t_s ≥ a_j·Kosten_s + b_j, minimiere Σ p_s t_s | Näherung; Stützstellen gleichmäßig in [60; 700] GE, nicht an ρ angepasst |
| **Zufall** (`evw_rng.py`) | ganzzahliger SplitMix64, Kopie der geprüften Vorlage des Portfolios (Hash im Test); die Zufallsprobleme brauchen nur Ganzzahlen und Gleichverteilung | kein Python-`hash()`, kein `random`, kein `numpy.random` in der Rechnung |

**Zufallsprobleme.** Je Problem vier Auszahlungen (G hoch, G niedrig, K hoch, K niedrig) ganzzahlig gleichverteilt in [−100; 200] GE, P(hoch) gleichverteilt in [0,1; 0,9], Testgüte q in [0,55; 0,95], Testkosten in [0; 10] GE (zufällig je Problem, oder in der App fest wie in Experiment 1); die Reihenfolge der Ziehungen steht in `evw_scenario.py`. Ein Problem ist **dominiert**, wenn eine Alternative in beiden Zuständen mindestens so gut ist wie die andere.
**Ein-Standort-Fall** (Experiment 4, Kosten): Kapazität 1 GE je Einheit, Zukauf 1,9 GE je Einheit bis 50 Einheiten, darüber Fehlmenge 3 GE je Einheit, lognormale Nachfrage mit Mittel 100 und Log-Streuung 0,35 in n gleich wahrscheinlichen Quantil-Mittelpunkten. Zahlen und Modell wie in `newsvendor-evpi-demo`, hier neu gerechnet.
**Vorgerechnet** (`evw_frozen.json`, erzeugt von `tools/evw_make_frozen.py`, Laufzeit 74 Sekunden): die Messreihen mit vielen LPs (vier Zufalls-Seeds mit je 1000 Zufallsproblemen, der Entscheidungswechsel mit 20 000 Problemen, die 25 Tangenten-LPs mit 2000 Szenarien). In der App steht „vorgerechnet“ daneben; `python tools/evw_make_frozen.py --check` rechnet die schnellen Teile neu und vergleicht (Zählungen als Band ± 2; `tests/test_frozen.py`).
**Laufzeit.** Der erste Aufruf der ganzen Seite braucht bei den Voreinstellungen etwa 2,6 Sekunden (davon rund 1,3 Sekunden für die Tabelle der Zustände); danach ist alles zwischengespeichert, und eine geänderte Eingabe rechnet nur ihre eigenen Teile. Die einzelnen Rechnungen am Reglerrand brauchen jeweils unter 1 Sekunde: Experiment 4 mit 500 Szenarien und 64 Tangenten etwa 0,7, 1000 Zufallsprobleme etwa 0,35, das gemischt-ganzzahlige Modell bei N = 8 bis etwa 0,85, der Entscheidungswechsel mit 20 000 Problemen etwa 0,45 Sekunden (Messung auf dem Entwicklungsrechner, einmal gemessen, Größenordnung, ohne das Zeichnen; Zeiten sind nie ein Test).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Kapazitätsfall (ausgedachte Zahlen), zwei Alternativen, zwei Zustände, sofern nicht anders genannt Sensitivität = Spezifität = q = 4/5, 5 GE je Test, Zufalls-Seed 20261120. Die Spalte „Test“ nennt die Testfunktion, die genau diese Form prüft (Bänder dort, wo Plattformen abweichen können; „Messung“ heißt: Zahl dieser README, im Test nur als Band oder nicht geprüft). Alle Zahlen dieser Tabelle stehen in `tests/test_claims.py` (Zahl → Test), soweit die Spalte einen Test nennt.

| Frage | Befund | Test |
|---|---|---|
| **B1** Wert ohne Test, EVPI, Nenner | EW(G) = 50, EW(K) = 30, Hellseher-Wert 115, EVPI = 65 GE = 56,5 % (13/23) von WS; als Anteil an RP wären es 130 % (65/50, irreführend, in der App nicht gezeigt) | `test_b1_ohne_test_evpi_nenner` |
| **B2** Ein Test (q = 4/5, ohne Kosten) | Wert 85, EVSI = 35 GE (7/13 von EVPI); wertlos für 1/2 ≤ q ≤ 17/30 (der Regler der App läuft von 1/2 bis 1); das Szenario-LP gibt 85 (4 Variablen, 2 Gleichungen) | `test_b2_ein_test_evsi` |
| **B3** Höchstens zwei Tests nacheinander, 5 GE je Test | Baum, Zustandsrechnung (6 Zustände, 7 Baum-Knoten) und gemischt-ganzzahliges Modell (10 Variablen, 3 binär, 6 Zeilen): 807/10 = 80,7 GE. Strategie: Wurzel testen; nach „+“ stoppen und G wählen; nach „−“ noch einmal testen. Feste Testzahl n = 0 / 1 / 2: 50 / 80 / 391/5 (= 78,2), das Szenario-LP gleich auf 1e-9; **Abstand zum Stoppen nach Beobachtung 7/10 GE** | `test_b3_handbeispiel_vier_verfahren` |
| **B4** Dreistufiger Fall (Kapazität, Nachfrage, Nachbesserung) | Baum über alle 17 Knickstellen in Brüchen und Szenario-LP mit nichtantizipativer Nachbesserung (22 Variablen, 9 Szenarien) geben 2117/16 = 132,3125 GE bei x₁ = 100 | `test_b4_dreistufig` |
| **B5** Wachstum bei N Tests nacheinander (q = 4/5, 5 GE) | Wert für N = 0 / 1 / 2 / 3 / 4 / 5 / 12: 50 / 80 / 80,7 / 87,8 / 88,024 / 90,296 / 91,434 GE (Baum gleich Zustandsrechnung in Brüchen, N = 0 bis 12); Knoten 2^(N+1) − 1 (8191 bei N = 12), Zustände (N+1)(N+2)/2 (91); Abstand zum besten festen n: 0 (N = 0, 1), 0,7 (N = 2), 3,4 (N = 3), 7,03 GE (N = 12, bestes festes n = 3 mit 84,4) | `test_b5_wachstum_n0_bis_12` |
| **B6** Szenario-LP bei fester Testzahl (n = 0 bis 4) gleich der Rechnung über die Verteilung der Glaubenszustände | in 5000 von 5000 LPs (1000 Zufallsprobleme je Zufalls-Seed, N = 4, vier Zufalls-Seeds: 20 000 von 20 000), größte Abweichung unter 2e-13; im Repo neu gerechnet in 300 LPs (60 Probleme) | `test_b6_lp_gleich_verteilungsrechnung` |
| **B7** Stoppen nach Beobachtung (1000 Zufallsprobleme, N = 4, Zufalls-Seed 20261120) | 516 nicht dominiert, in 252 lohnt sich ein Test, **in 220 ist Stoppen besser als jede feste Testzahl** (Lücke über 1e-7 GE; alle 220 unter den 252; der Hauptlauf hat unter den vier Zufalls-Seeds die höchste Zahl, siehe B8), in den übrigen 32 lohnenden Problemen reicht eine feste Zahl. Lücke unter den 252: Mittel 2,17 GE, Median 1,91, größte 9,46; Mittel über alle 1000: 0,55 GE. Baum gleich Zustandsrechnung in 1000 von 1000 | `test_b7_zufall_hauptmessung_eingefroren`, `test_b7_zufall_neu_gerechnet_als_band` |
| **B8** Gegenprobe, drei weitere Zufalls-Seeds (1, 777, 424242; je 1000 Probleme) | Stoppen besser in 199 / 196 / 188 von 1000, ein Test lohnt in 232 / 231 / 212; Szenario-LP gleich der Verteilungsrechnung in allen LPs. Die Zahl heißt also **188 bis 220 von 1000** (vier Zufalls-Seeds), nicht eine Stelle genau | `test_b8_vier_zufalls_seeds` |
| **B9** Gemischt-ganzzahliges Modell gegen Baum | in 300 von 300 Problemen je Zufalls-Seed (N = 4, `mip_rel_gap` = 1e-9, Abstand höchstens 1,1e-6, das ist die Ganzzahligkeitstoleranz); im Repo in 40 Problemen und im Handbeispiel bis N = 8 | `test_b9_mip_gleich_baum` |
| **B10** Standardlücke von HiGHS | Mit der Standardeinstellung (relative Lücke 1e-4) wich das Modell in 2 von 300 Problemen (Zufalls-Seed 777) um mehr als 1e-5 GE und bis zu 0,00085 GE unter den Baumwert ab (mit der Schwelle 1e-6 sind es 23 von 300): eine Toleranz der Software, kein Befund über das Modell | Messung (im Test nur als Band 0 bis 10) |
| **B11** Zustände gegen Knoten (exakt gezählt, Brüche) | N = 2 / 4 / 8 / 14: Baum-Knoten 7 / 31 / 511 / 32 767; gleiche Tests 6 / 15 / 45 / 120 = (N+1)(N+2)/2 (alle N von 0 bis 14); zwei Sorten abwechselnd (4/5, 3/5) 7 / 22 / 95 / 372 (Formel Σ (⌈j/2⌉+1)(⌊j/2⌋+1) für alle N bis 14); Quotenverhältnisse, die multiplikativ unabhängig sind (Primzahl-Konstruktion: der i-te Test hat die i-te Primzahl als Quotenverhältnis) 7 / 31 / 511 / 32 767 = alle 2^(N+1) − 1; bei bloß paarweise verschiedenen Verhältnissen können Folgen zusammenfallen (Gegenbeispiel 2, 3, 6 bei N = 3: 14 statt 15 Zustände, je Tiefe 1 / 2 / 4 / 7 statt 1 / 2 / 4 / 8) | `test_b11_zustaende_gegen_knoten` |
| **B12** Entscheidungswechsel bei Exponentialnutzen (9843 nicht dominierte Zufallsprobleme aus 20 000, Zufalls-Seed 20261121, ein Schritt ohne Test) | andere Wahl als beim Erwartungswert in 33,1 / 22,9 / 13,6 / 5,2 % bei ρ = 20 / 50 / 100 / 300 GE; mittlerer Verlust in Sicherheitsäquivalent-GE in diesen Problemen 35,2 / 26,6 / 16,9 / 6,2 (größter 189,9 / 137,1 / 82,2 / 24,8). Gegenprobe ρ = 50 mit den Zufalls-Seeds 1 / 777 / 424242: 22,3 / 22,0 / 22,1 % | `test_b12_entscheidungswechsel` |
| **B13** Tangenten-LP, Ein-Standort-Fall (2000 Szenarien) | beste Kapazität der 2000-Punkte-Verteilung x(ρ) = 100,923 / 108,953 / 130,856 / 189,836 / 242,346 für ρ = 500 / 200 / 100 / 50 / 25. Mehrverlust (GE, exakt bewertet) bei **64 Tangenten höchstens 0,13** in allen fünf Zeilen (bei ρ = 50: 0,047 GE von einem Sicherheitsäquivalent von 209,8 GE, 0,02 %), bei 4 Tangenten bei ρ = 50 **19,1 GE**; der Mehrverlust fällt **nicht monoton** in m (ρ = 25: 1,15 bei m = 8, 2,08 bei m = 16; ρ = 200: 0,0007 bei m = 8, 0,0116 bei m = 32) | `test_b13_tangenten_eingefroren`, `test_b13_tangenten_neu_gerechnet_m4_m8` |
| **B14** Szenarienzahl und beste Kapazität (neu gemessen) | Die exakt beste Kapazität bei ρ = 50 hängt an der Zahl der Quantil-Mittelpunkte: 172,4 bei 500, 181,1 bei 1000, 189,8 bei 2000 Szenarien; das Tangenten-LP mit 64 Tangenten hat bei 500 Szenarien und ρ = 50 einen Mehrverlust von 0,0095 GE. Die exakt beste Kapazität ist eine Eigenschaft der Verteilung aus n Punkten, nicht der stetigen Lognormalverteilung: dafür existiert das Sicherheitsäquivalent der Kosten nicht (das Exponentialmoment ist unendlich, weil die Kosten linear mit der Nachfrage wachsen), und die beste Kapazität wächst in den gemessenen Zahlen (500, 1000, 2000 Punkte) mit n weiter | `test_b14_szenarienzahl` |
| **B15** Nenner im Ein-Standort-Fall (Kosten, geschlossene Form) | x_RP = 96,790, RP = 129,1122, EEV = 129,2444, WS = 100; EVPI = 29,1122, VSS = 0,1322 GE: EVPI **22,55 % von RP, 29,11 % von WS, 22,52 % von EEV**; VSS 0,102 % / 0,132 % / 0,102 %. Die README von `newsvendor-evpi-demo` nennt EVPI 22,5 % und VSS 0,10 % (Nenner RP): bestätigt | `test_b15_nenner_ein_standort` |
| **B16** Nenner in einem Handbeispiel mit Kosten (drei Szenarien 50 / 100 / 200 mit ¼, ½, ¼, Kapazität 1 GE, Fehlmenge 3 GE) | RP = 175 bei x = 100, WS = 225/2, EEV = 1425/8; EVPI = 125/2 = 62,5 (5/14 = 35,7 % von RP, 5/9 = 55,6 % von WS), VSS = 25/8 = 3,125 (1/56 = 1,79 % von RP); das Szenario-LP gibt RP = 175 | `test_b16_nenner_handbeispiel_kosten` |
| **B17** Wegweiser | 72 Kombinationen der fünf Merkmale, jede mit Voraussetzung, Belegzeile und Gegenfall. Die Regel „Stoppen erlaubt: Baum, Zustandsrechnung oder gemischt-ganzzahliges Modell“ überempfiehlt in 780 von 1000 Zufallsproblemen (748 ohne lohnenden Test, 32 mit ausreichender fester Testzahl): eine Vorsichtsregel, kein Aufwandsurteil für den Einzelfall | `test_b17_wegweiser_zaehlung`, `tests/test_wegweiser.py` |

### Hinweise zu den Zahlen

- **Plattform.** Die vorgerechneten Zählungen (`evw_frozen.json`) stammen von Python 3.14.6, numpy 2.5.3, scipy 1.18.1 unter Windows. Zählungen über Gleitkomma-Vergleiche gelten als Band (± 2 Probleme, Anteile ± 0,01), LP-Werte auf 1e-7 bis 1e-9, MIP-Werte auf 1e-5, nie bitgleich. Was in Brüchen gerechnet ist (B1 bis B5, B11, B16), ist auf jeder Plattform gleich und wird exakt verglichen. Der Linux-Lauf steht unter „Verifikation“.
- **Gleichstände.** HiGHS darf bei Gleichstand eine andere Lösung (Ecke) wählen: verglichen werden nur Werte, nie Strategien des LP. Beim Entscheidungswechsel (B12) zählen Probleme, in denen eine der beiden Wahlen auf einem Gleichstand unter 1e-9 beruht, nicht mit (im Hauptlauf ändert das keine Zählung).
- **Zählweise der Anteile.** „220 von 1000“ ist eine Zählung über die Zufallsprobleme dieses Zufalls-Seeds; die Spanne 188 bis 220 über vier Zufalls-Seeds zeigt, dass die Zahl nicht auf die Einerstelle genau ist. Der Anteil gilt für diese Verteilung der Zufallsprobleme, nicht für alle Probleme.
- **Zeiten.** Rechenzeiten (Sekunden, Laufzeit der Vorrechnung) sind Messungen dieser README auf dem Entwicklungsrechner, einmal gemessen und Größenordnung; kein Test prüft sie.
- **Ursachen.** Warum das Stoppen in manchen Problemen mehr bringt (der Baum kann nach dem Ergebnis auf die Folge reagieren, ein fester Plan nicht), ist eine Deutung aus der Rechnung, keine eigene Messung.

### Befunde und Korrekturen gegenüber dem Plan

- **Gegenprobe des Entscheidungswechsels (B12).** Die frühere Messung nannte 22,7 / 22,9 / 22,1 % bei ρ = 50 für drei weitere Zufalls-Seeds (Seeds und Problemzahl dort nicht festgehalten); die Neumessung mit den Zufalls-Seeds 1, 777 und 424242, je 20 000 Problemen und dem Ausschluss von Gleichständen gibt **22,3 / 22,0 / 22,1 %**. Der Hauptlauf (22,9 %) bleibt unverändert.
- **MIP mit Lückentoleranz in allen vier Zufalls-Seeds (B9).** Früher war das Modell mit 1e-9 nur an zwei Läufen neu gerechnet; jetzt in 300 von 300 Problemen je Zufalls-Seed, alle vier. Die Standardlücke (B10) wurde nur für Zufalls-Seed 777 gezählt.
- **MIP-Obergrenze in der App.** Gemessen am Handbeispiel (q = 4/5 und q = 11/20, 5 GE): bis N = 8 unter 1 Sekunde, bei N = 9 und 10 bis etwa 3 Sekunden; darum rechnet die App das Modell bis N = 8. Bei N = 10 weicht der MIP-Wert um bis zu 1,4e-6 vom Baum ab (Ganzzahligkeitstoleranz).
- **Szenarienzahl (B14).** Dass die exakt beste Kapazität bei kleinem ρ so stark an der Zahl der Quantil-Mittelpunkte hängt, stand nicht in der früheren Messung; die Messwerte aus 2000 Szenarien gelten für 2000 Szenarien. Bei 500 Szenarien läuft das LP mit 64 Tangenten in etwa 0,7 Sekunden (früher nicht gemessen), darum wird es live gerechnet; die Szenarienzahl 2000 ist vorgerechnet (bis 5 Sekunden je LP).
- **22,52 statt 22,53 %.** EVPI/EEV ist 22,5249 %, gerundet 22,52 (die frühere Messung nannte 22,53).
- **Wegweiser.** Die Entscheidungshilfe fragt fünf Merkmale ab (Informationsstruktur, Nutzen, Zustandsstruktur, Zufallsquelle, Gilt der Erwartungsnutzen); die zuerst vorgesehene Zeile je Nachbar-Demo steht nicht als Tabelle in der App, sondern als Verweis in „Wo die Annahmen enden“. Die Zeile zum Einflussdiagramm fehlt („nicht gemessen“).
- **Gestrichen.** Nichts aus den fünf Experimenten; hinzugekommen ist der dreistufige Fall als Teil des Ergebnisses (B4) und eine Tabelle der Glaubenszustände je Tiefe (bis N = 5).

## Quellen

Nur so zitiert, wie gelesen (Wortlaut = Volltext gelesen; Kapitelliste = nur Inhaltsangabe):
- **Rollback.** Raiffa, Schlaifer, *Applied Statistical Decision Theory*, Harvard 1961, Abschn. 1.2.1, S. 7–8: Mitteln über die Ergebnisse und Zurückfalten von den Blättern (Wortlaut gelesen). Die Rechnung dieser Demo ist eigene Nachrechnung.
- **Glaubenszustand.** Kaelbling, Littman, Cassandra, *Planning and Acting in Partially Observable Stochastic Domains*, Artificial Intelligence 101 (1998), 99–134: der Glaubenszustand ist eine hinreichende Statistik für Vorgeschichte und Anfangsglauben; die exakte Lösung wächst exponentiell (Wortlaut gelesen). Hier nur die Rechenregel nach Bayes.
- **Zustandszusammenlegung.** Sonnenberg, Beck, *Markov Models in Medical Decision Making*, Med Decis Making 13 (1993), 322–338: die Markov-Annahme der Zusammenlegung sei in der Praxis nicht streng erfüllt, aber nötig (Wortlaut gelesen).
- **Zweistufige stochastische Programme.** Birge, Louveaux, *Introduction to Stochastic Programming*: nur die Kapitelliste und das GAMS-Farmmodell der Universität KTH gelesen, das Buch nicht.
- **Nicht gelesen:** Howard 1966, Pratt, Savage im Original (kein Zugriff). Keine Aussage dieser Demo hängt an ihnen.
- Eine Neuheitsaussage gibt es nicht.

## Ehrliche Grenzen

- **Ein Fall, zwei Alternativen, zwei Zustände.** Mehr Alternativen, mehr Zustände, Tests mit unsymmetrischer Güte in der App und die **Wahl zwischen verschiedenen Tests** (Informationskauf) sind nicht gemessen; die Orakel-Tests prüfen unsymmetrische Tests im Szenario-LP gegen die Verteilungsrechnung, mehr nicht.
- **Zufallsprobleme sind ausgedacht.** Alle Anteile (220 von 1000, 22,9 %) gelten für diese Verteilung der Zufallsprobleme; eine andere Verteilung der Auszahlungen, Wahrscheinlichkeiten oder Testgüten kann andere Anteile geben. Gemessen ist nur der Fall gleicher Tests nacheinander.
- **Das gemischt-ganzzahlige Modell** ist an Zufallsproblemen nur bei N = 4 geprüft (300 Probleme je Zufalls-Seed), im Handbeispiel bis N = 8; größere N sind nicht gemessen.
- **Der Exponentialnutzen** ist nur für die Wahl ohne Test (Experiment 4, ein Schritt) und im Ein-Standort-Fall gezählt, nicht für das Stoppen nach Beobachtung. Die Stützstellen des Tangenten-LP liegen gleichmäßig in [60; 700] GE; eine an ρ angepasste Platzierung ist nicht gemessen. Bei Potenznutzen hängt die Wahl am Vermögen: nicht gemessen.
- **Der Wegweiser** fasst die Messungen dieser Demo zusammen und beweist keine Regel für andere Aufgaben. Die Regel „Stoppen erlaubt: Baum, Zustandsrechnung oder MIP“ überempfiehlt in den meisten Zufallsproblemen (B17).
- **Die Primzahl-Zählung** (alle Zustände verschieden) ist eine Konstruktion, kein Befund über „zufällige“ Tests. Ein früherer Versuch mit q_i = (2i+3)/(2i+5) zählte bei N = 7 nur 247 von 255 Zuständen (Zufallstreffer der Brüche).
- **Nicht gemessen:** Zeit in Sekunden für größere Bäume als N = 14, Stichprobenfehler im Szenario-LP (Nachbar-Demo `stichprobenbaum-demo`), Allais-Struktur (Nachbar-Demo `erwartungsnutzen-grenze-demo`), CVaR. Rechenzeiten und der Linux-Lauf stehen unter „Hinweise zu den Zahlen“ und „Verifikation“.

## Verifikation

- **Handwerte in Brüchen** (`tests/test_kern.py`): jede Zahl des Handbeispiels als Rechnung von Hand im Test (50, 65, 85, 35, 807/10, 50 / 80 / 391/5, 7/10, 6 gegen 7 Zustände, 2117/16).
- **Orakel-Tests** (unabhängige Rechenwege): `tests/test_oracle_stoppregeln.py` zählt **alle Stoppregeln** auf (jede Teilmenge der inneren Knoten als „hier stoppen“, bis 2^15 = 32 768 Regeln bei N = 4) und vergleicht das Maximum mit dem Baum; Zustandsrechnung gegen Baum in Brüchen (N = 0 bis 10); feste Testzahl gegen die **Binomialsumme** über die Zahl der positiven Ergebnisse. `tests/test_oracle_szenario_lp.py`: Szenario-LP gegen Verteilungsrechnung (300 LPs, auch bei unsymmetrischen Tests), MIP gegen Baum, dreistufig Baum in Brüchen gegen LP und gegen ein dichtes Raster. `tests/test_oracle_exponential.py`: Zwei-Punkt-Sicherheitsäquivalent (0 oder 100, ½, ρ = 100: 37,988549304), Normalverteilung μ − σ²/(2ρ), Delta-Eigenschaft, Ein-Standort-Fall geschlossen gegen LP gegen Baum über die Szenarien, Tangenten unter der Kurve und gleich an den Stützstellen, x(ρ) gegen die früheren Werte auf 0,002.
- **Eigenschaften** (`tests/test_eigenschaften.py`, je 40 Zufallsprobleme): Umbenennen der Alternativen, Verschieben und Skalieren, adaptiv mindestens so gut wie bestes festes n, mehr erlaubte Tests schaden nie, EVSI höchstens EVPI, bei q = 1/2 jeder Test wertlos, bei Kosten ab EVPI nie getestet.
- **Randfälle** (`tests/test_edge_cases.py`): N = 0, q = 1 (kein Teilen durch 0), q = 1/2, P(hoch) = 0 und 1, Gleichstand, dominierte Alternative, N = 14, sehr kleine Risikotoleranz.
- **Eingefrorenes** (`tests/test_frozen.py`): Aufbau, innere Stimmigkeit, `evw_make_frozen.py --check`. **Zufall** (`tests/test_rng.py`): Hash der Vorlage, Referenzfolge, keine anderen Zufallsquellen.
- **Wegweiser** (`tests/test_wegweiser.py`): Wahrheitstabelle über alle 72 Kombinationen, jede Ausgabe mit Voraussetzung und existierender Belegzeile, Größe nicht als Eingang, „nicht gemessen“ und „keines der drei“. **Verdikte** (`tests/test_verdicts.py`) an handgesetzten Werten, **Presets** (`tests/test_presets.py`): je Preset die erwartete Verdikt-Zeile.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, `tests/test_app.py`), Layout (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Änderung eines Reglers hebt die Markierung auf; `tests/test_layout.py`), Footer, Laufzeit-Importe. Im Browser durchgeklickt (alle sechs Presets, kein Skriptfehler, KaTeX ohne Fehler); **nicht mit echtem Mausklick getestet:** das Hervorheben der Knoten im Diagramm beim Darüberfahren.
- **Linux-Lauf** (Docker, Python 3.12 wie die CI, frische Installation, `tools/demo_linux_check.py` des Website-Repos; Stand des Baus, 2026-11-17): alle 164 Tests grün, 66 Sekunden.
- **Testzahl:** 164 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis (Experiment 1), Kernfragen, Experimente 2 bis 5, Grenzen, Formeln |
| `evw_kern.py` | Kern: Glaubenszustand, Baum, Zustandsrechnung, feste Testzahl, Szenario-LP, gemischt-ganzzahliges Modell, dreistufiger Fall |
| `evw_exponential.py` | Sicherheitsäquivalent, Tangenten-LP, Ein-Standort-Fall, Entscheidungswechsel |
| `evw_scenario.py` | Zufallsprobleme (Reihenfolge der Ziehungen, Dominanz) |
| `evw_wegweiser.py` | Wegweiser: Wahrheitstabelle, Belegzeilen, Gegenfälle |
| `evw_evaluation.py` | Experimente, Kennzahlen, Verdikte (reine Funktionen) |
| `evw_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `evw_constants.py`, `evw_presets.py`, `evw_layout.py` | Regler-Grenzen und Presets, Permalink und Schnellstart, fester Kopf |
| `evw_rng.py` | SplitMix64 (unveränderte Kopie der geprüften Vorlage) |
| `evw_frozen.json` | vorgerechnete Messreihen (siehe Methodik) |
| `tools/evw_make_frozen.py` | erzeugt und prüft `evw_frozen.json` |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, scipy, plotly |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Handwerte, drei Orakel-Dateien, Eigenschaften, Randfälle, Zustände, Zufall, Wegweiser, Eingefrorenes, Verdikte, Presets, Claims, AppTest, Layout, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Informationskauf** (Wahl zwischen verschiedenen Tests, 2^T Läufe oder ein MIP): nicht gerechnet; das gemischt-ganzzahlige Modell wählt nur zwischen Stoppen und Weitertesten.
- **CVaR, Potenznutzen, Allais, Stichprobenfehler:** Verweise auf Nachbar-Demos, keine eigene Rechnung.
- **Einflussdiagramm** (Zeile im Wegweiser): „nicht gemessen“, keine Empfehlung.
- **PDF-Export** der Ergebnisse: nicht umgesetzt.
- **Tabelle „Verfahren je Nachbar-Demo“** im Wegweiser: nur als Verweise in „Wo die Annahmen enden“.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Entscheidungsanalyse: Information, Risiko, Grenzen](https://sebastianhanisch.net/konzepte-entscheidungsanalyse.html).

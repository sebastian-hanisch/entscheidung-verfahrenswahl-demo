"""Verfahrenswahl: Baum, Szenario-LP oder Zustandsrechnung – interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Dieselbe Entscheidung unter Unsicherheit (ausgedachter Kapazitätsfall, ein Modell, kein Fall) in vier Verfahren gerechnet: wo sie dasselbe geben, wo das Szenario-LP aufhört
und wann die Zustandsrechnung wirklich spart. Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import json
from fractions import Fraction
from pathlib import Path

import streamlit as st

import evw_constants as C
import evw_evaluation as ev
import evw_kern as K
import evw_wegweiser as W
from evw_layout import dunkles_design, kopf_css
from evw_presets import (
    SETTING_SPECS,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from evw_visualization import (
    build_baum,
    build_dreistufig,
    build_luecken,
    build_mehrverlust,
    build_tangenten,
    build_verfahren,
    build_wachstum,
    build_wechsel,
)

st.set_page_config(page_title="Verfahrenswahl – Sebastian Hanisch", layout="wide")

FROZEN_PFAD = Path(__file__).resolve().parent / "evw_frozen.json"


@st.cache_data(show_spinner=False)
def _frozen():
    with open(FROZEN_PFAD, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _tests(n, q20, c, stoppen):
    return ev.analyse_tests(n, q20, c, stoppen)


@st.cache_data(show_spinner=False)
def _dreistufig(x1):
    return ev.analyse_dreistufig(x1)


@st.cache_data(show_spinner=False)
def _zufall(seed, m, n, art, q20, c):
    return ev.zufall_lauf(seed, m, n, art, q20, c)


@st.cache_data(show_spinner=False)
def _detail(seed, m, n, art, q20, c):
    z = ev.zufall_lauf(seed, m, n, art, q20, c)
    return ev.detail_problem(z.probleme[z.index_max], n), z.probleme[z.index_max]


@st.cache_data(show_spinner=False)
def _zustaende():
    return ev.zustaende_tabellen()


@st.cache_data(show_spinner=False)
def _glauben(art, n):
    return ev.glaubenszustaende_je_tiefe(art, n)


@st.cache_data(show_spinner=False)
def _tangenten(rho, m, szen):
    return ev.tangenten_ergebnis(rho, m, szen, _frozen())


@st.cache_data(show_spinner=False)
def _wechsel(seed, m, rho):
    return ev.wechsel_live(seed, m, rho)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in evw_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Verfahrenswahl: Baum, Szenario-LP oder Zustandsrechnung")
st.markdown(
    """
Eine Entscheidung unter Unsicherheit lässt sich auf mehreren Wegen rechnen: als **Entscheidungsbaum (Ereignisknoten)** mit Rollback, als **Zustandsrechnung** (derselbe Rollback mit Gedächtnis),
als **Szenario-LP** und, wenn nach jedem Ergebnis gestoppt werden darf, als **gemischt-ganzzahliges Modell**. Auf einem **ausgedachten Kapazitätsfall** (die Auszahlungen sind erfunden: ein Modell, kein Fall)
geben alle vier im Wert dasselbe. Die Demo zeigt, **wo das stimmt, wo es aufhört und was es kostet**: der Schluss „Baum gleich LP, also egal“ gilt bei fester Informationsstruktur und Erwartungswert; zwei Fälle, in denen er endet, zeigen die Experimente 2 und 4.

Weiter unten: die **Kernfragen**, drei weitere **🔬 Experimente**, der **🧰 Wegweiser** (Experiment 5), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So rechnen die vier Verfahren", expanded=True):
    st.markdown(
        """
**Der Fall (ausgedachte Zahlen, in Geldeinheiten GE).** Alternative **G** (groß bauen) zahlt bei hoher Nachfrage +200 GE und bei niedriger −100 GE, Alternative **K** (klein bauen oder mieten) sicher +30 GE; hohe Nachfrage hat die
Wahrscheinlichkeit 1/2. Ein **Test** meldet „+“ (hoch) oder „−“ mit Sensitivität = Spezifität = q und kostet c GE; bis zu N Tests nacheinander, **nach jedem Ergebnis darf gestoppt werden**.
Der **Glaubenszustand** b = P(hoch) wird nach Bayes fortgeschrieben; nach „+“ gilt b′ = b·q / (b·q + (1 − b)(1 − q)).

1. **Entscheidungsbaum (Ereignisknoten):** Rollback von den Blättern her, jeder Knoten einzeln gerechnet (kein Gedächtnis). Gezählt wird ein **Historienknoten** je Folge von Testergebnissen der Länge höchstens N: 2^(N+1) − 1 Knoten.
2. **Zustandsrechnung:** dieselbe Rückwärtsrechnung, aber mit Gedächtnis auf dem Paar (Tiefe, Glaubenszustand): zwei Folgen mit demselben Bruch sind **ein** Zustand.
3. **Szenario-LP:** je Szenario (Zustand der Nachfrage, Folge der Ergebnisse) ein Anteil x der Alternative G; **Nichtantizipativität** heißt: Szenarien mit gleicher Ergebnisfolge teilen sich denselben Plan (eine Gleichung je Folge).
   Es gibt **einen Lauf je Informationsstruktur**, also je feste Testzahl.
4. **Gemischt-ganzzahliges Modell (MIP):** ein Wert je Historienknoten und eine Binärvariable „jetzt entscheiden“ je innerem Knoten; damit lässt sich das Stoppen nach Beobachtung in einem Modell ausdrücken.

**Abgrenzung.** „Entscheidungsbaum“ meint hier das Rechenverfahren mit Ereignisknoten; der Baum des Verfahrens CART (maschinelles Lernen) ist etwas anderes, die Demo nennt ihre Entscheidungshilfe **Wegweiser**. Die Nachbar-Demo
[`entscheiden-lernen-wahl-demo`](https://github.com/sebastian-hanisch/entscheiden-lernen-wahl-demo) wählt Verfahren für das Entscheiden aus Daten; diese Demo vergleicht Rechenverfahren für ein gegebenes Entscheidungsmodell.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Zufallsprobleme**")
    st.number_input("Zufalls-Seed der Zufallsprobleme", 0, C.SEED_MAX, key="seed_input", step=1,
                    help="Gilt für die Experimente 2, 4 und 5. Der Entscheidungswechsel in Experiment 4 benutzt Zufalls-Seed + 1 (voreingestellt 20261121).")
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

frozen = _frozen()

# =====================================================================================================================================================
# Ergebnis (Experiment 1): dieselbe Entscheidung in vier Verfahren
# =====================================================================================================================================================
st.markdown("## 🎯 Ergebnis (Experiment 1): dieselbe Entscheidung in vier Verfahren")
st.caption("Kapazitätsfall mit ausgedachten Auszahlungen (G: +200 oder −100 GE, K: +30 GE, P(hoch) = 1/2). Alles Folgende gilt für diesen Fall mit zwei Alternativen und zwei Zuständen.")
r1, r2, r3, r4 = st.columns(4)
with r1:
    st.slider("Zahl der Tests N (höchstens)", C.E1_N_MIN, C.E1_N_MAX, key="e1_n", help="Bis zu N Tests nacheinander; bei „Stoppen“ darf nach jedem Ergebnis aufgehört werden.")
with r2:
    st.select_slider("Testgüte q (Sensitivität = Spezifität)", options=list(range(C.E1_Q20_MIN, C.E1_Q20_MAX + 1)), key="e1_q20",
                     format_func=lambda v: f"{v}/20 = {v / 20:.2f}".replace(".", ","), help="Raster 1/20, damit alles in Brüchen rechenbar bleibt; q = 1/2 ist ein wertloser Test.")
with r3:
    st.slider("Testkosten c (GE je Test)", C.E1_C_MIN, C.E1_C_MAX, step=C.E1_C_STEP, key="e1_c", help="Kosten jedes einzelnen Tests in Geldeinheiten.")
with r4:
    st.radio("Informationsstruktur", C.STOPPEN_OPTIONEN, key="e1_stoppen", help="„Stoppen erlaubt“: nach jedem Ergebnis wird neu entschieden. „Feste Testzahl“: die Zahl der Tests steht vorab fest.")

v1 = _tests(int(ss["e1_n"]), int(ss["e1_q20"]), float(ss["e1_c"]), ss["e1_stoppen"] == C.STOPPEN_JA)
ev0 = v1.ohne_test
m1, m2, m3, m4 = st.columns(4)
m1.metric("Wert mit Test (Baum)" if v1.stoppen else "Wert (Baum ohne Stoppen)", f"{ev.de_kurz(v1.baum_wert, 3)} GE",
          delta=f"{ev.de_kurz(v1.abstand, 3)} gegen bestes festes n" if v1.stoppen else "ohne Stoppen: gleich", delta_color="normal" if v1.stoppen else "off")
m2.metric(f"bestes Szenario-LP (feste Testzahl n = {v1.bestes_n})", f"{ev.de_kurz(v1.lp[0], 3)} GE", delta="Referenz", delta_color="off")
m3.metric("EVPI (Wert vollkommener Information)", f"{ev.wert_text(ev0['evpi'])} GE",
          delta=f"{ev.pct(ev0['evpi_von_ws'])} von WS", delta_color="off",
          help="RP = Wert ohne Information (50 GE), WS = Wert mit vollkommener Information (115 GE). Als Anteil an RP wäre EVPI im Gewinnfall 130 % (irreführend); die Demo zeigt hier GE und den Anteil an WS und nennt den Nenner.")
m4.metric("EVSI (ein Test ohne Kosten)", f"{ev.de_kurz(v1.evsi_ein_test, 3)} GE", delta=f"{ev.pct(v1.evsi_ein_test / ev0['evpi'])} von EVPI", delta_color="off",
          help="Wert der Information eines einzelnen Tests der gewählten Güte q, ohne Kosten.")

st.dataframe(ev.zeilen_verfahren(v1), hide_index=True, width="stretch")
balken = [("Baum", float(v1.baum_wert), C.COLOR_BAUM), ("Zustandsrechnung", float(v1.dp_wert), C.COLOR_ZUSTAND)]
if v1.mip is not None:
    balken.append(("MIP", float(v1.mip[0]), "#8e44ad"))
balken.append((f"Szenario-LP (n = {v1.bestes_n})", float(v1.lp[0]), C.COLOR_LP))
st.plotly_chart(build_verfahren(balken), width="stretch", key="chart_verfahren")

if v1.stoppen:
    st.markdown("**Strategie des Baums** (nach „+“ meldet der Test hohe Nachfrage; bei Gleichstand wird gestoppt):")
    st.code("\n".join("    " * d + t for d, t in K.strategie_text(v1.strategie)), language=None)
    if v1.n_max <= C.BAUM_BILD_N_MAX:
        knoten = ev.baum_info(Fraction(1, 2), K.A_CAP, v1.c, K.tests_gleich(v1.q, v1.n_max))
        st.plotly_chart(build_baum(knoten, v1.n_max), width="stretch", key="chart_baum")
        st.caption("Grün: die Strategie stoppt hier, blau: sie testet weiter, grau: wird nie erreicht. Je Knoten: Glaubenszustand b und Wert in GE; Einzelheiten beim Darüberfahren.")
    else:
        st.caption(f"Das Diagramm des Baums gibt es bis N = {C.BAUM_BILD_N_MAX}; darüber hat der Baum {K.baum_knoten(v1.n_max)[0]} Knoten.")
else:
    st.caption(f"Bei fester Testzahl steht die Zahl der Tests vorab fest: hier ist n = {v1.bestes_n} die beste Wahl aus 0 bis {v1.n_max}. Werte je Testzahl: "
               + ", ".join(f"n = {n}: {ev.de_kurz(w, 3)}" for n, w in enumerate(v1.fest_werte)) + " GE.")
kind, text = ev.verdict_verfahren(v1)
getattr(st, kind)(text)

st.markdown("#### Dreistufiger Fall (Kapazität, Nachfrage, Nachbesserung)")
st.markdown(
    "Zum Vergleich ein Fall ohne Tests: die **erste Kapazität x₁** wird vor der Nachfrage gebaut (1 GE je Einheit), die Nachfrage D₁ ist 70, 100 oder 150 (Wahrscheinlichkeiten 0,3 / 0,4 / 0,3); danach darf nachgebessert "
    "werden (1,5 GE je Einheit), und die zweite Nachfrage ist D₁ mal 0,8, 1 oder 1,25 (¼ / ½ / ¼); Zukauf kostet 1,9 GE je Einheit bis 50 Einheiten, jede weitere fehlende Einheit 3 GE. Ausgedachte Zahlen."
)
st.slider("Erste Kapazität x₁", C.E1_X1_MIN, C.E1_X1_MAX, step=C.E1_X1_STEP, key="e1_x1", help="Die erwarteten Kosten des Baums bei dieser ersten Kapazität; das Optimum liegt bei 100.")
d3 = _dreistufig(int(ss["e1_x1"]))
st.plotly_chart(build_dreistufig(d3.raster, d3.x1, d3.optimum, d3.x1_optimum), width="stretch", key="chart_dreistufig")
kind, text = ev.verdict_dreistufig(d3)
getattr(st, kind)(text)

st.markdown("---")

# =====================================================================================================================================================
# Kernfragen (live, aus den Experimenten; gefüllt, sobald die Rechnungen vorliegen)
# =====================================================================================================================================================
kern = st.container()

seed = int(ss["seed_input"])
z2 = _zufall(seed, int(ss["e2_m"]), int(ss["e2_n"]), ss["e2_art"], int(ss["e1_q20"]), float(ss["e1_c"]))
tab = _zustaende()
with kern:
    st.markdown("## ❓ Kernfragen")
    antworten = ev.kernfragen(v1, z2, int(ss["e3_n"]), ss["e3_art"], tab)
    fragen = ("Geben Baum, Szenario-LP und Zustandsrechnung dasselbe?", "Wo reicht das Szenario-LP nicht?", "Wann spart die Zustandsrechnung wirklich?")
    for frage, (art, text) in zip(fragen, antworten):
        st.markdown(f"**{frage}**")
        getattr(st, art)(text)

st.markdown("---")

# =====================================================================================================================================================
# Experiment 2: Zufallsprobleme
# =====================================================================================================================================================
st.subheader("🔬 Experiment 2: Wo hört das Szenario-LP auf? Stoppen nach Beobachtung")
st.markdown(
    "Zufallsprobleme mit **zwei Alternativen** (Auszahlungen je Zustand ganzzahlig gleichverteilt in [−100; 200] GE) und **zwei Zuständen**, P(hoch) gleichverteilt in [0,1; 0,9]. Je Problem werden der Baum, die Zustandsrechnung "
    "und das **beste Szenario-LP mit fester Testzahl** (n = 0 bis N) verglichen: darf nach jedem Test gestoppt werden, kann ein fester Plan zu wenig. Die Zufallsprobleme sind ausgedacht, kein Fall."
)
e2a, e2b, e2c = st.columns(3)
with e2a:
    st.slider("Zahl der Tests N (höchstens)", C.E2_N_MIN, C.E2_N_MAX, key="e2_n", help="Bis zu N gleiche Tests nacheinander; das MIP ist bis N = 4 an Zufallsproblemen gemessen.")
with e2b:
    st.select_slider("Zahl der Zufallsprobleme", options=list(C.E2_M_OPTIONEN), key="e2_m", help="Mehr Probleme, glattere Zählungen; die vorgerechnete Messung hat 1000.")
with e2c:
    st.radio("Testgüte und Kosten", C.E2_ART_OPTIONEN, key="e2_art", help="Zufällig: q gleichverteilt in [0,55; 0,95], Kosten in [0; 10] GE je Problem. Sonst gelten q und c aus Experiment 1 für alle Probleme.")
z2 = _zufall(seed, int(ss["e2_m"]), int(ss["e2_n"]), ss["e2_art"], int(ss["e1_q20"]), float(ss["e1_c"]))
n_lohnt = z2.test_lohnt
x1c, x2c, x3c, x4c = st.columns(4)
x1c.metric("nicht dominierte Probleme", f"{z2.nicht_dominiert} von {z2.m}", help="Keine Alternative ist in beiden Zuständen mindestens so gut wie die andere.")
x2c.metric("Test lohnt sich", f"{z2.test_lohnt} von {z2.m}", help="Der Baumwert liegt über dem Wert ohne Test.")
x3c.metric("Stoppen besser als jedes feste n", f"{z2.adaptiv_besser} von {z2.m}", delta=f"{ev.pct(z2.adaptiv_besser / z2.m)}", delta_color="off", help="Lücke über 1e-7 GE.")
x4c.metric("mittlere Lücke bei lohnendem Test", f"{ev.de(float(sum(z2.luecken_lohnend) / len(z2.luecken_lohnend)) if z2.luecken_lohnend else 0.0, 2)} GE",
           delta=f"größte {ev.de(max(z2.luecken_lohnend) if z2.luecken_lohnend else 0.0, 2)} GE", delta_color="off")
if z2.luecken_lohnend:
    cl, cr = st.columns(2)
    with cl:
        st.plotly_chart(build_luecken(z2.luecken), width="stretch", key="chart_luecken")
        st.caption("Nur Probleme mit positiver Lücke (über 1e-7 GE).")
    with cr:
        st.plotly_chart(build_verfahren([("Baum (Stoppen erlaubt)", z2.mittel_baum_lohnend, C.COLOR_BAUM), ("bestes Szenario-LP (feste Zahl)", z2.mittel_fest_lohnend, C.COLOR_LP)]),
                        width="stretch", key="chart_zufall_balken")
        st.caption("Mittlerer Wert in den Problemen, in denen sich ein Test lohnt.")
st.dataframe(
    [{"Zufalls-Seed": s, "Probleme": z["m"], "nicht dominiert": z["nicht_dominiert"], "Test lohnt": z["test_lohnt"], "Stoppen besser": z["adaptiv_besser"],
      "mittlere Lücke (lohnend, GE)": ev.de(z["luecke_mittel_lohnend"], 2), "LP gleich Rechnung": f"{z['lp_geprueft'] - z['lp_abweichungen']} von {z['lp_geprueft']}",
      "MIP gleich Baum": f"{z['mip_geprueft'] - z['mip_abweichungen']} von {z['mip_geprueft']}"} for s, z in frozen["zufall"].items()],
    hide_index=True, width="stretch")
st.caption("Vorgerechnet, nicht live (`tools/evw_make_frozen.py`): 1000 Zufallsprobleme je Zufalls-Seed, N = 4, zufällige Testgüte und Kosten; Szenario-LP bei jeder festen Testzahl 0 bis 4 und das MIP an den ersten 300 Problemen.")
if z2.test_lohnt:
    p_detail = _detail(seed, int(ss["e2_m"]), int(ss["e2_n"]), ss["e2_art"], int(ss["e1_q20"]), float(ss["e1_c"]))
    det, pr = p_detail
    with st.expander("Das Problem mit der größten Lücke, in allen Verfahren (live)"):
        st.markdown(
            f"Auszahlungen G = ({pr.alts[0][0]}; {pr.alts[0][1]}) GE, K = ({pr.alts[1][0]}; {pr.alts[1][1]}) GE (hoch; niedrig), P(hoch) = {ev.de(pr.b0, 3)}, q = {ev.de(pr.q, 3)}, c = {ev.de(pr.c, 2)} GE, "
            f"bis zu {z2.n_tests} Tests. Baum: **{ev.de(det['baum'], 3)} GE**; bestes festes n: {ev.de(max(det['fest']), 3)} GE (Szenario-LP je n: "
            + ", ".join(f"n = {n}: {ev.de(w, 3)}" for n, w in enumerate(det["lp"])) + ")"
            + (f"; gemischt-ganzzahliges Modell: {ev.de(det['mip'], 3)} GE." if det["mip"] is not None else ".")
        )
        st.code("\n".join("    " * d + t for d, t in K.strategie_text(det["strategie"])), language=None)
kind, text = ev.verdict_zufall(z2, frozen)
getattr(st, kind)(text)

st.markdown("---")

# =====================================================================================================================================================
# Experiment 3: Knoten gegen Zustände
# =====================================================================================================================================================
st.subheader("🔬 Experiment 3: Wann spart die Zustandsrechnung wirklich?")
st.markdown(
    "Die Zustandsrechnung legt zusammen, was dieselbe Zukunft hat: zwei Ergebnisfolgen, die auf **denselben Glaubenszustand** führen. Bei **gleichen Tests** tun das sehr viele (+ − und − + enden beide bei 1/2); "
    "bei Tests, deren Quotenverhältnisse **multiplikativ unabhängig** sind (hier die Primzahlen), führt jede Folge zu einem eigenen Bruch, und keine Folgen werden zusammengelegt; bei bloß verschiedenen Quotenverhältnissen (etwa 2, 3, 6) können Folgen zusammenfallen. Gezählt werden die verschiedenen Glaubenszustände als Brüche, exakt."
)
e3a, e3b = st.columns(2)
with e3a:
    st.slider("Zahl der Tests N", C.E3_N_MIN, C.E3_N_MAX, key="e3_n", help="Bis N = 14: 32 767 Baum-Knoten.")
with e3b:
    st.radio("Testart", C.E3_ART_OPTIONEN, key="e3_art", help="Gleiche Tests: q = 4/5. Zwei Sorten: 4/5 und 3/5 abwechselnd. Paarweise verschieden: Quotenverhältnis q/(1 − q) des i-ten Tests ist die i-te Primzahl (eine Konstruktion, kein Zufall; die Quotenverhältnisse sind dadurch multiplikativ unabhängig).")
n3, art3 = int(ss["e3_n"]), ss["e3_art"]
kn, zs = K.baum_knoten(n3)[0], tab[art3]["summe"][n3]
s1, s2, s3 = st.columns(3)
s1.metric("Baum-Knoten", ev.tausender(kn))
s2.metric("Zustände der Zustandsrechnung", ev.tausender(zs), delta=f"{ev.pct(zs / kn)} der Knoten", delta_color="off")
s3.metric("Formel", ev.tausender(K.zustaende_formel(art3, n3)), delta="stimmt mit der Zählung überein" if K.zustaende_formel(art3, n3) == zs else "weicht ab", delta_color="off",
          help="Gleiche Tests: (N+1)(N+2)/2. Zwei Sorten: Summe über j von (⌈j/2⌉+1)(⌊j/2⌋+1). Paarweise verschieden: 2^(N+1) − 1.")
st.plotly_chart(build_wachstum(tab, art3, n3), width="stretch", key="chart_wachstum")
if n3 <= C.E3_TABELLE_N_MAX:
    g = _glauben(art3, n3)
    st.markdown("**Die Glaubenszustände je Tiefe** (Brüche; ein Zustand ist ein Paar aus Tiefe und Glaubenszustand):")
    st.dataframe([{"Tiefe": t, "Zahl": len(z), "Glaubenszustände b = P(hoch)": ", ".join(z)} for t, z in enumerate(g)], hide_index=True, width="stretch")
else:
    st.caption(f"Die Tabelle der Glaubenszustände je Tiefe gibt es bis N = {C.E3_TABELLE_N_MAX}.")
st.caption("Verweis: Rückwärtsrechnung auf dem Zustand „Bestandsposition“ statt dem Glaubenszustand zeigt [`bestell-rueckwaerts-demo`](https://github.com/sebastian-hanisch/bestell-rueckwaerts-demo); "
           "dort fällt der Zustand zusammen, weil die Bestandsposition die Vorgeschichte zusammenfasst.")
kind, text = ev.verdict_zustaende(n3, art3, tab)
getattr(st, kind)(text)

st.markdown("---")

# =====================================================================================================================================================
# Experiment 4: Exponentialnutzen im LP
# =====================================================================================================================================================
st.subheader("🔬 Experiment 4: Der Exponentialnutzen im LP")
st.markdown(
    "Ein Szenario-LP rechnet mit dem **Erwartungswert**. Wer nach dem **Sicherheitsäquivalent** (Exponentialnutzen, Risikotoleranz ρ) wählt, wählt in einem Teil der Probleme etwas anderes. Der Exponentialnutzen selbst ist "
    "nichtlinear, **exakt** also nicht im LP darstellbar; **stückweise linear** (m Tangenten an die Kurve) schon näherungsweise. Gerechnet am **Ein-Standort-Fall** (Kosten: Kapazität 1 GE je Einheit, Zukauf 1,9 GE bis 50 Einheiten, "
    "Fehlmenge 3 GE, lognormale Nachfrage mit Mittel 100; Zahlen wie in [`newsvendor-evpi-demo`](https://github.com/sebastian-hanisch/newsvendor-evpi-demo), hier neu gerechnet). Die „beste Kapazität“ gilt für die Verteilung aus n gleich wahrscheinlichen Punkten, nicht für die stetige Lognormalverteilung: für sie gibt es kein Sicherheitsäquivalent der Kosten (das Exponentialmoment ist unendlich), und die beste Kapazität wächst in den gemessenen Zahlen mit n weiter."
)
e4a, e4b, e4c, e4d = st.columns(4)
with e4a:
    st.select_slider("Risikotoleranz ρ (GE)", options=list(C.E4_RHO_OPTIONEN), key="e4_rho", help="Kleiner heißt risikoscheuer.")
with e4b:
    st.select_slider("Zahl der Tangenten m", options=list(C.E4_M_OPTIONEN), key="e4_m", help="Stützstellen gleichmäßig in [60; 700] GE.")
with e4c:
    st.select_slider("Zahl der Szenarien", options=list(C.E4_SZEN_OPTIONEN), key="e4_szen", help="250 und 500 werden live gerechnet; 2000 ist vorgerechnet (das LP braucht dort bis zu 5 Sekunden).")
with e4d:
    st.select_slider("Zufallsprobleme (Entscheidungswechsel)", options=list(C.E4_PROBLEME_OPTIONEN), key="e4_probleme", help="Zahl der Zufallsprobleme aus Experiment 2 (ein Schritt, ohne Test) für den Anteil „andere Wahl“.")
rho, m4v, szen = int(ss["e4_rho"]), int(ss["e4_m"]), int(ss["e4_szen"])
with st.spinner("Tangenten-LP wird gelöst …" if szen != C.E4_SZEN_VORGERECHNET else "Tabelle wird gelesen …"):
    t4 = _tangenten(rho, m4v, szen)
w4 = _wechsel(seed + C.E4_SEED_VERSATZ, int(ss["e4_probleme"]), rho)
f1, f2, f3, f4 = st.columns(4)
f1.metric("beste Kapazität (exakt)", ev.de(t4["x_exakt"], 1), help="Skalare Suche über das Sicherheitsäquivalent der Kosten; exakt beste Kapazität der Verteilung aus n Punkten, nicht der stetigen Lognormalverteilung.")
f2.metric("Kapazität aus dem Tangenten-LP", ev.de(t4["x_lp"], 1), delta=f"{ev.de(t4['x_lp'] - t4['x_exakt'], 1)} gegen exakt", delta_color="off")
f3.metric("Mehrverlust (GE)", ev.de(t4["mehrverlust"], 3), delta="vorgerechnet" if not t4["live"] else "live gerechnet", delta_color="off",
          help="Sicherheitsäquivalent der Kapazität aus dem LP minus das der exakt besten Kapazität, jeweils exakt bewertet.")
anteil = w4["rho"][float(rho)]["wechsel"] / w4["nicht_dominiert"] if w4["nicht_dominiert"] else 0.0
f4.metric("andere Wahl als bei Erwartungswert", ev.pct(anteil), delta=f"{w4['rho'][float(rho)]['wechsel']} von {w4['nicht_dominiert']}", delta_color="off")
cc, cd = st.columns(2)
ce_ref = t4["ce_exakt"]
with cc:
    c_, exakt, huelle, stuetz, u_st = ev.kurve_tangenten(float(rho), m4v, ce_ref)
    st.plotly_chart(build_tangenten(c_, exakt, huelle, stuetz, u_st, rho, m4v), width="stretch", key="chart_tangenten")
    st.caption("Die Hülle der Tangenten liegt nie über der Kurve (die Kurve ist konvex) und trifft sie an den Stützstellen.")
with cd:
    st.plotly_chart(build_mehrverlust(frozen["tangenten"]["zeilen"], rho, m4v, t4["mehrverlust"] if t4["live"] and szen != C.E4_SZEN_VORGERECHNET else None), width="stretch", key="chart_mehrverlust")
    st.caption("Vorgerechnet mit 2000 Szenarien: der Mehrverlust fällt nicht gleichmäßig mit m (Deutung, nicht gemessen: Die Stützstellen liegen relativ zu ρ mal näher, mal weiter vom besten Wert). Das gewählte Ergebnis mit anderer Szenarienzahl (live) liegt nicht auf diesen Kurven.")
wl, wr = st.columns([1, 1])
with wl:
    st.plotly_chart(build_wechsel({r: v["wechsel"] / w4["nicht_dominiert"] if w4["nicht_dominiert"] else 0.0 for r, v in w4["rho"].items()}), width="stretch", key="chart_wechsel")
with wr:
    st.markdown(
        f"**Entscheidungswechsel.** In {w4['nicht_dominiert']} nicht dominierten Zufallsproblemen (Zufalls-Seed {w4['seed']}, {w4['m']} Probleme) wählt das risikoneutrale LP in dem links gezeigten Anteil eine andere Alternative "
        "als das Sicherheitsäquivalent. Das ist ein Schritt ohne Test; die Zahl gilt für diese Zufallsprobleme, nicht für alle."
    )
    st.dataframe([{"ρ (GE)": int(r), "andere Wahl": v["wechsel"], "Anteil": ev.pct(v["wechsel"] / w4["nicht_dominiert"] if w4["nicht_dominiert"] else 0.0),
                   "mittlerer Verlust (GE)": ev.de(v["verlust_mittel"], 1), "größter Verlust (GE)": ev.de(v["verlust_max"], 1)} for r, v in w4["rho"].items()], hide_index=True, width="stretch")
kind, text = ev.verdict_exponential(rho, m4v, t4, w4, frozen)
getattr(st, kind)(text)

st.markdown("---")

# =====================================================================================================================================================
# Experiment 5: Wegweiser
# =====================================================================================================================================================
st.subheader("🧰 Experiment 5: Der Wegweiser, geprüft")
st.markdown(
    "**Formfrage zuerst:** welches Verfahren passt, hängt an der Form der Aufgabe, nicht an ihrer Größe. Der Wegweiser fasst die Messungen dieser Demo zusammen; jede Ausgabe nennt ihre **Voraussetzung**, eine **Belegzeile** und einen **Gegenfall**. "
    "Wo nichts gemessen ist, steht „nicht gemessen“ und keine Empfehlung. Er ist eine Zusammenfassung der Messungen dieser Demo, kein Beweis einer Regel für andere Aufgaben."
)
g1, g2, g3, g4, g5 = st.columns(5)
with g1:
    st.selectbox("Informationsstruktur", W.INFO, key="e5_info")
with g2:
    st.selectbox("Nutzen", W.NUTZEN, key="e5_nutzen")
with g3:
    st.selectbox("Zustandsstruktur", W.ZUSTAND, key="e5_zustand")
with g4:
    st.selectbox("Zufallsquelle", W.QUELLE, key="e5_quelle")
with g5:
    st.selectbox("Gilt der Erwartungsnutzen?", W.EU, key="e5_eu")
emp = W.wegweiser(ss["e5_info"], ss["e5_nutzen"], ss["e5_zustand"], ss["e5_quelle"], ss["e5_eu"])
if emp.art == W.ART_KEINES:
    st.warning(f"**{emp.verfahren}.**")
elif emp.art == W.ART_NICHT_GEMESSEN:
    st.info("**Nicht gemessen.** Keine Empfehlung.")
else:
    st.success(f"**Empfehlung:** {emp.verfahren}.")
st.markdown(f"**Voraussetzung:** {emp.voraussetzung}")
for zeile in W.belegzeilen(emp, frozen):
    st.markdown(f"**Belegzeile:** {zeile}")
st.markdown(f"**Gegenfall:** {emp.gegenfall}")
pruefung = ev.wahlregel_pruefung(z2)
kind, text = ev.verdict_wegweiser(emp, pruefung)
getattr(st, kind)(text)

st.markdown("---")

# =====================================================================================================================================================
# Grenzen
# =====================================================================================================================================================
st.subheader("🚧 Wo die Annahmen enden")
zh = frozen["zufall"]["20261120"]
gl = frozen["wechsel"]
t64 = next(r for r in frozen["tangenten"]["zeilen"] if r["rho"] == 50.0 and r["m"] == 64)
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Beleg (diese Demo) |
|---|---|---|
| **Feste Informationsstruktur**: „Baum gleich Szenario-LP“ | Darf nach jedem Test gestoppt werden, ist das beste Szenario-LP mit fester Testzahl in {zh['adaptiv_besser']} von {zh['m']} Zufallsproblemen schlechter als der Baum | Experiment 2 (vorgerechnet) |
| **Erwartungswert** als Maß | Das risikoneutrale LP wählt bei ρ = 50 in {ev.pct(gl['rho']['50']['wechsel'] / gl['nicht_dominiert'])} der nicht dominierten Zufallsprobleme anders; ein Tangenten-LP mit 64 Tangenten erreicht {ev.de(t64['mehrverlust'], 2)} GE Mehrverlust | Experiment 4 |
| **Gleiche Zustände fallen zusammen** | Bei N = 14 Tests mit multiplikativ unabhängigen Quotenverhältnissen (Primzahl-Konstruktion) hat die Zustandsrechnung {ev.tausender(tab[K.TESTARTEN[2]]['summe'][14])} Zustände, so viele wie der Baum Knoten; bei gleichen Tests {ev.tausender(tab[K.TESTARTEN[0]]['summe'][14])} | Experiment 3 |
| **Die Verteilung ist bekannt** | Aus einer Stichprobe kommt ein zusätzlicher Fehler, den diese Demo nicht misst | Verweis [`stichprobenbaum-demo`](https://github.com/sebastian-hanisch/stichprobenbaum-demo) |
| **Der Erwartungsnutzen gilt** | Bei Allais-artigen Wahlen gibt es keine Nutzenfunktion, die beide Wahlen erklärt; kein Verfahren dieser Demo hilft | Verweis [`erwartungsnutzen-grenze-demo`](https://github.com/sebastian-hanisch/erwartungsnutzen-grenze-demo) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind zwei Alternativen mit Auszahlungen $h_a$ (hohe) und $l_a$ (niedrige Nachfrage), die Anfangswahrscheinlichkeit $b_0 = P(\text{hoch})$, die Testgüte $q$ (Sensitivität = Spezifität), die Testkosten $c$ und die
höchste Zahl $N$ der Tests nacheinander. Gesucht ist der Wert der besten Strategie und die Strategie selbst.

| Symbol | Bedeutung |
|---|---|
| $b$ | Glaubenszustand: Wahrscheinlichkeit der hohen Nachfrage nach den bisherigen Ergebnissen |
| $h_a,\ l_a$ | Auszahlung der Alternative $a$ (G oder K) bei hoher, bei niedriger Nachfrage |
| $q,\ c,\ N$ | Güte eines Tests, Kosten je Test (GE), höchste Zahl der Tests |
| $s(b)$ | Wert, jetzt zu entscheiden |
| $P(+),\ P(-)$ | Wahrscheinlichkeit des Ergebnisses „+“ bzw. „−“ beim nächsten Test |
| $V_j(b)$ | Wert im Knoten der Tiefe $j$ mit Glaubenszustand $b$ |
| $x_{\theta f}$ | Anteil der Alternative G im Szenario (Zustand $\theta$, Ergebnisfolge $f$) |
| $z_h,\ v_h,\ M$ | Binärvariable „jetzt entscheiden“, Wert im Historienknoten $h$, große Konstante |
| $\rho$ | Risikotoleranz in GE |

**Bayes.** Nach „+“ beziehungsweise „−“:

$$b^{+}=\frac{b\,q}{b\,q+(1-b)(1-q)},\qquad P(+)=b\,q+(1-b)(1-q)$$

$$b^{-}=\frac{b\,(1-q)}{b\,(1-q)+(1-b)\,q},\qquad P(-)=b\,(1-q)+(1-b)\,q$$

**Wert ohne weiteren Test.**

$$s(b)=\max_{a}\ \bigl(b\,h_a+(1-b)\,l_a\bigr)$$

**Rollback (Baum) und Zustandsrechnung.** Beide lösen dieselbe Gleichung; der Baum rechnet jeden Historienknoten einzeln, die Zustandsrechnung speichert $V_j(b)$ je Paar $(j,b)$:

$$V_N(b)=s(b),\qquad V_j(b)=\max\Bigl\{\,s(b),\ -c+P(+)\,V_{j+1}(b^{+})+P(-)\,V_{j+1}(b^{-})\Bigr\}$$

Bei Gleichstand wird gestoppt. Zahl der Knoten und Zustände:

$$2^{N+1}-1\ \text{Historienknoten},\qquad \frac{(N+1)(N+2)}{2}\ \text{Zustände bei gleichen Tests}$$

**Feste Testzahl und Szenario-LP.** Bei genau $n$ Tests ohne Stoppen ist der Wert die Summe über alle Ergebnisfolgen $f$ der Länge $n$:

$$W_n=-n\,c+\sum_{f}P(f)\,s(b_f)$$

Das Szenario-LP hat je Szenario $(\theta,f)$ mit Wahrscheinlichkeit $\pi(\theta,f)$ einen Anteil $x_{\theta f}\in[0,1]$ der Alternative G:

$$\max\ \sum_{\theta,f}\pi(\theta,f)\Bigl(a^{K}_{\theta}+x_{\theta f}\,(a^{G}_{\theta}-a^{K}_{\theta})\Bigr)-n\,c$$

Die **Nichtantizipativität** verlangt, dass der Plan nur von dem abhängt, was bis dahin beobachtet wurde:

$$x_{\text{hoch},f}=x_{\text{niedrig},f}\qquad\text{für jede Ergebnisfolge }f$$

Eine feste Testzahl ist hier Teil der Informationsstruktur: ein Lauf je $n$.

**Gemischt-ganzzahliges Modell des Stoppens.** Je Historienknoten $h$ eine Wertvariable $v_h$, je innerem Knoten eine Binärvariable $z_h$ (1 = jetzt entscheiden); in den Blättern gilt $v_h=s(b_h)$. Maximiert wird $v_0$ unter

$$v_h\le s(b_h)+M\,(1-z_h)$$

$$v_h\le -c+P_h(+)\,v_{h+}+P_h(-)\,v_{h-}+M\,z_h,\qquad z_h\in\{0,1\}$$

Die Software braucht eine relative Lückentoleranz; die Demo setzt sie auf $10^{-9}$ (Standard von HiGHS: $10^{-4}$).

**Exponentialnutzen.** Bei Gewinnen $g_\theta$ mit Wahrscheinlichkeiten $p_\theta$ ist das Sicherheitsäquivalent

$$\mathrm{CE}=-\rho\,\ln\Bigl(\sum_{\theta}p_\theta\,e^{-g_\theta/\rho}\Bigr)$$

bei Kosten $C_s$ in gleich wahrscheinlichen Szenarien $\mathrm{CE}=\rho\ln\bigl(\tfrac1n\sum_s e^{C_s/\rho}\bigr)$. Das Tangenten-LP ersetzt $u(C)=e^{(C-c_{\mathrm{ref}})/\rho}$ durch die Hülle von $m$ Tangenten in den Stützstellen $c_j$:

$$\min\ \frac1n\sum_s t_s\qquad\text{mit}\qquad t_s\ge a_j\,C_s+b_j,\quad a_j=\frac{u(c_j)}{\rho},\quad b_j=u(c_j)-a_j\,c_j$$

wobei $C_s$ die Kosten im Szenario $s$ sind (Kapazität, Zukauf und Fehlmenge als Variablen).

**Wert der Information.** Mit RP als Wert ohne Information und WS als Wert mit vollkommener Information gilt im Kapazitätsfall (Gewinne): $\mathrm{EVPI}=\mathrm{WS}-\mathrm{RP}$; bei Kosten $\mathrm{EVPI}=\mathrm{RP}-\mathrm{WS}$. Die Demo nennt neben jedem Anteil den Nenner.

**Quellen** (so gelesen, wie angegeben). Rollback („Mitteln und Zurückfalten“): Raiffa, Schlaifer, *Applied Statistical Decision Theory*, Harvard 1961, Abschn. 1.2.1, S. 7–8 (Wortlaut gelesen). Der Glaubenszustand als hinreichende Statistik
und das exponentielle Wachstum der exakten Lösung: Kaelbling, Littman, Cassandra, *Planning and Acting in Partially Observable Stochastic Domains*, Artificial Intelligence 101 (1998), 99–134 (Wortlaut gelesen). Die Markov-Annahme der
Zustandszusammenlegung sei in der Praxis nicht streng erfüllt, aber nötig: Sonnenberg, Beck, Med Decis Making 13 (1993), 322–338 (Wortlaut gelesen). Zweistufige stochastische Programme und Szenario-LP: Birge, Louveaux, *Introduction to Stochastic Programming*
(Kapitelliste und das GAMS-Farmmodell der Universität KTH gelesen, das Buch nicht). Alle Formeln und Zahlen dieser Demo sind eigene Nachrechnung in diesem Modell; eine Neuheit wird nicht behauptet.
        """
    )

sync_query_params({k: ss[k] for k in SETTING_SPECS})

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Entscheidungsanalyse: Information, Risiko, Grenzen](https://sebastianhanisch.net/konzepte-entscheidungsanalyse.html)."
)

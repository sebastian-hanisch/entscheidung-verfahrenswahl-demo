"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import evw_constants as C
import evw_kern as K


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25) if legend else None, plot_bgcolor="rgba(0,0,0,0)", showlegend=legend)
    return lock_axes(fig)


def _bruch(b):
    return f"{b.numerator}/{b.denominator}" if b.denominator != 1 else str(b.numerator)


def build_verfahren(zeilen_werte):
    """Waagerechte Balken: Wert je Verfahren (GE). `zeilen_werte` = Liste (Name, Wert, Farbe)."""
    namen = [z[0] for z in zeilen_werte][::-1]
    werte = [z[1] for z in zeilen_werte][::-1]
    farben = [z[2] for z in zeilen_werte][::-1]
    fig = go.Figure(go.Bar(x=werte, y=namen, orientation="h", marker_color=farben, text=[f"{w:.3f}".replace(".", ",") for w in werte], textposition="outside",
                           hovertemplate="%{y}: %{x:.4f} GE<extra></extra>"))
    lo, hi = min(werte), max(werte)
    spanne = max(hi - lo, 1e-9)
    fig.update_xaxes(title_text="Wert (GE), höher ist besser", range=[lo - 4 * spanne - 1, hi + 2 * spanne + 1])
    return _base(fig, height=240, legend=False)


def build_baum(knoten, n_tests):
    """Der Baum bis N = 3: Knoten nach Tiefe und Blattreihenfolge, die Strategie farbig (grün: stoppt, blau: weiter testen), nicht erreichte Knoten grau."""
    n_blaetter = 2 ** n_tests
    pos = {}
    for k in knoten:
        if k["tiefe"] == n_tests:
            index = int("".join("0" if ch == "+" else "1" for ch in k["pfad"]) or "0", 2)
            pos[k["pfad"]] = (k["tiefe"], n_blaetter - 1 - index)
    for tiefe in range(n_tests - 1, -1, -1):
        for k in knoten:
            if k["tiefe"] == tiefe:
                ys = [pos[k["pfad"] + "+"][1], pos[k["pfad"] + "−"][1]]
                pos[k["pfad"]] = (tiefe, sum(ys) / 2)
    fig = go.Figure()
    for k in knoten:
        if k["kinder"]:
            for kind in k["kinder"]:
                x0, y0 = pos[k["pfad"]]
                x1, y1 = pos[kind["pfad"]]
                fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color="rgba(130,130,130,0.5)", width=1.5), hoverinfo="skip", showlegend=False))
    xs, ys, farben, texte, hover = [], [], [], [], []
    for k in knoten:
        x, y = pos[k["pfad"]]
        xs.append(x)
        ys.append(y)
        if not k["erreichbar"]:
            farben.append(C.COLOR_GRAU)
        elif k["stoppt"]:
            farben.append(C.COLOR_BAUM)
        else:
            farben.append(C.COLOR_LP)
        aktion = f"stoppen: {K.ALT_KURZ[k['wahl']]}" if k["stoppt"] else "noch ein Test"
        texte.append(f"b = {_bruch(k['b'])}<br>{float(k['wert']):.4g}".replace(".", ","))
        weiter = "—" if k["weiter"] is None else f"{float(k['weiter']):.4g}"
        hover.append(f"Folge „{k['pfad'] or 'Start'}“<br>b = {_bruch(k['b'])}<br>Stopp-Wert {float(k['stopp']):.4g}, Weiter-Wert {weiter}<br>Wert {float(k['wert']):.4g}<br>{aktion}")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=texte, textposition="top center", marker=dict(size=14, color=farben, line=dict(width=1, color="white")),
                             hovertext=hover, hoverinfo="text", showlegend=False, textfont=dict(size=10)))
    fig.update_xaxes(title_text="Zahl der bisherigen Tests (Tiefe)", dtick=1, range=[-0.4, n_tests + 0.5])
    fig.update_yaxes(visible=False, range=[-0.8, n_blaetter - 0.2])
    return _base(fig, height=max(300, 120 * n_blaetter), legend=False)


def build_dreistufig(raster, x1, optimum, x_opt):
    """Erwartete Kosten über der ersten Kapazität x1 (Baum, Rollback in Brüchen); markiert: gewähltes x1 und das Optimum."""
    xs = [r[0] for r in raster]
    ys = [r[1] for r in raster]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=C.COLOR_BAUM, width=3), name="Baum (Rollback)", hovertemplate="x₁ = %{x}: %{y:.3f} GE<extra></extra>"))
    wert_x1 = next((r[1] for r in raster if r[0] == x1), None)
    if wert_x1 is not None:
        fig.add_trace(go.Scatter(x=[x1], y=[wert_x1], mode="markers", marker=dict(size=13, color=C.COLOR_ZUSTAND), name="gewähltes x₁", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[float(x_opt)], y=[float(optimum)], mode="markers", marker=dict(size=11, symbol="diamond", color=C.COLOR_LP), name="Optimum (Baum = Szenario-LP)", hoverinfo="skip"))
    fig.update_xaxes(title_text="erste Kapazität x₁")
    fig.update_yaxes(title_text="erwartete Kosten (GE)")
    return _base(fig, height=320)


def build_luecken(luecken, grenze=1e-7):
    """Verteilung der Lücke (Baum minus bestes festes n) über die Probleme, in denen sie positiv ist."""
    positive = [x for x in luecken if x > grenze]
    fig = go.Figure(go.Histogram(x=positive, nbinsx=20, marker_color=C.COLOR_BAUM, hovertemplate="Lücke %{x} GE: %{y} Probleme<extra></extra>"))
    fig.update_xaxes(title_text="Lücke (Baum minus bestes festes n), GE")
    fig.update_yaxes(title_text="Zahl der Probleme")
    return _base(fig, height=300, legend=False)


def build_wachstum(tab, art_gewaehlt, n_gewaehlt):
    """Baum-Knoten und Zustände gegen N (logarithmische Achse), je Testart eine Kurve; der gewählte Punkt ist hervorgehoben."""
    ns = list(range(0, C.E3_N_MAX + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[K.baum_knoten(n)[0] for n in ns], mode="lines+markers", line=dict(color="rgba(120,120,120,0.9)", dash="dot", width=2), name="Baum-Knoten 2^(N+1) − 1",
                             hovertemplate="N = %{x}: %{y} Knoten<extra></extra>"))
    for art in K.TESTARTEN:
        breit = 4 if art == art_gewaehlt else 2
        fig.add_trace(go.Scatter(x=ns, y=tab[art]["summe"], mode="lines+markers", line=dict(color=C.COLOR_ARTEN[art], width=breit), name=f"Zustände: {art}",
                                 hovertemplate="N = %{x}: %{y} Zustände<extra></extra>"))
    fig.add_vline(x=n_gewaehlt, line=dict(color="rgba(120,120,120,0.5)", width=1))
    fig.update_xaxes(title_text="Zahl der Tests N", dtick=1)
    fig.update_yaxes(title_text="Anzahl (logarithmisch)", type="log")
    return _base(fig, height=380)


def build_tangenten(c, exakt, huelle, stuetz, u_stuetz, rho, m):
    """Der Exponentialnutzen und seine Näherung durch m Tangenten (nur die Stützstellen im Fenster)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c, y=exakt, mode="lines", line=dict(color=C.COLOR_BAUM, width=3), name="exp((c − cₑ)/ρ)", hovertemplate="c = %{x:.1f}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=c, y=huelle, mode="lines", line=dict(color=C.COLOR_LP, width=2, dash="dash"), name=f"Tangenten-Hülle ({m})", hovertemplate="c = %{x:.1f}: %{y:.3f}<extra></extra>"))
    if len(stuetz):
        fig.add_trace(go.Scatter(x=stuetz, y=u_stuetz, mode="markers", marker=dict(size=8, color=C.COLOR_ZUSTAND), name="Stützstellen", hoverinfo="skip"))
    fig.update_xaxes(title_text="Kosten c (GE)")
    fig.update_yaxes(title_text=f"Nutzen (relativ zu cₑ), ρ = {rho}")
    return _base(fig, height=340)


def build_mehrverlust(zeilen, rho_gewaehlt, m_gewaehlt, live_punkt=None):
    """Mehrverlust des Tangenten-LP gegen die Zahl der Tangenten (logarithmische Achse; je ρ eine Kurve), vorgerechnet mit 2000 Szenarien. Nicht monoton in m."""
    fig = go.Figure()
    for rho in sorted({r["rho"] for r in zeilen}, reverse=True):
        z = [r for r in zeilen if r["rho"] == rho]
        breit = 4 if int(rho) == rho_gewaehlt else 2
        fig.add_trace(go.Scatter(x=[r["m"] for r in z], y=[max(r["mehrverlust"], 1e-6) for r in z], mode="lines+markers", line=dict(width=breit), name=f"ρ = {int(rho)}",
                                 hovertemplate="m = %{x}: Mehrverlust %{y:.4f} GE<extra></extra>"))
    if live_punkt is not None:
        fig.add_trace(go.Scatter(x=[m_gewaehlt], y=[max(live_punkt, 1e-6)], mode="markers", marker=dict(size=13, symbol="x", color="black"), name="gewählt (live)", hoverinfo="skip"))
    fig.update_xaxes(title_text="Zahl der Tangenten m", type="log", tickvals=list(C.E4_M_OPTIONEN), ticktext=[str(m) for m in C.E4_M_OPTIONEN])
    fig.update_yaxes(title_text="Mehrverlust (GE, logarithmisch)", type="log")
    return _base(fig, height=340)


def build_wechsel(anteile):
    """Anteil der nicht dominierten Zufallsprobleme, in denen das risikoneutrale LP anders wählt als das Sicherheitsäquivalent, je ρ."""
    rhos = [str(int(r)) for r in anteile]
    fig = go.Figure(go.Bar(x=rhos, y=[100 * a for a in anteile.values()], marker_color=C.COLOR_ZUSTAND, text=[f"{100 * a:.1f} %".replace(".", ",") for a in anteile.values()],
                           textposition="outside", hovertemplate="ρ = %{x}: %{y:.1f} %<extra></extra>"))
    fig.update_xaxes(title_text="Risikotoleranz ρ (GE)", type="category")
    fig.update_yaxes(title_text="andere Wahl (%)", rangemode="tozero")
    return _base(fig, height=300, legend=False)

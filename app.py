import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

from nba_api.stats.static import teams
from nba_api.stats.endpoints import (
    commonteamroster,
    playergamelog,
    scoreboardv2,
    leaguedashplayerstats
)


# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="NBA Analyzer", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      h1 {letter-spacing: -0.5px;}
      .stCaption {opacity: 0.85;}
      div[data-testid="stMetric"]{
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 12px 12px;
        background: rgba(255,255,255,0.03);
      }
      .chart-card{
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 12px 12px 2px 12px;
        background: rgba(255,255,255,0.02);
      }
      .cal-card{
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 12px 12px;
        background: rgba(255,255,255,0.02);
      }
      .mini-muted {opacity: 0.85; font-size: 12px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("NBA Player Analyzer 🏀")
st.caption("pa las jubiladoras by GV")


# -------------------------
# ALTAIR THEME (dark)
# -------------------------
def _altair_theme():
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "axis": {
                "labelFont": "Inter",
                "titleFont": "Inter",
                "labelFontSize": 11,
                "titleFontSize": 12,
                "gridColor": "#2A2F3A",
                "domainColor": "#2A2F3A",
                "tickColor": "#2A2F3A",
                "labelColor": "#C9D1D9",
                "titleColor": "#C9D1D9",
            },
            "title": {"font": "Inter", "color": "#E6EDF3", "fontSize": 14}
        }
    }

try:
    alt.themes.register("nba_theme", _altair_theme)
except Exception:
    pass
alt.themes.enable("nba_theme")


# -------------------------
# SEASON HELPERS
# -------------------------
def current_season_str():
    today = datetime.today()
    year, month = today.year, today.month
    start_year = year if month >= 10 else year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


# -------------------------
# CACHE HELPERS
# -------------------------
@st.cache_data(show_spinner=False)
def get_teams_df():
    df = pd.DataFrame(teams.get_teams()).sort_values("full_name").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def team_maps():
    """TEAM_ID -> (abbrev, full_name)"""
    tdf = pd.DataFrame(teams.get_teams())
    id2abbr = dict(zip(tdf["id"].astype(int), tdf["abbreviation"].astype(str)))
    id2name = dict(zip(tdf["id"].astype(int), tdf["full_name"].astype(str)))
    return id2abbr, id2name


@st.cache_data(show_spinner=True, ttl=60 * 60)
def fetch_team_roster(team_id: int, season: str):
    ro = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
    df = ro.get_data_frames()[0]
    if "PLAYER" not in df.columns or "PLAYER_ID" not in df.columns:
        return pd.DataFrame(columns=["PLAYER", "PLAYER_ID"])
    df = df[["PLAYER", "PLAYER_ID", "NUM", "POSITION", "HEIGHT", "WEIGHT", "AGE", "EXP", "SCHOOL"]].copy()
    return df.sort_values("PLAYER").reset_index(drop=True)


@st.cache_data(show_spinner=True, ttl=60 * 30)
def fetch_gamelog(player_id: int, season: str, season_type: str):
    gl = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star=season_type
    )
    df = gl.get_data_frames()[0]

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    df["Local"] = df["MATCHUP"].str.contains("vs")
    df["Rival"] = df["MATCHUP"].str.split().str[-1]

    keep = ["GAME_DATE", "Rival", "Local", "WL", "MIN", "PTS", "REB", "AST", "FG3M"]
    df = df[keep].rename(columns={"FG3M": "3PM"})

    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df["PA"]  = df["PTS"] + df["AST"]
    df["PR"]  = df["PTS"] + df["REB"]

    return df


# -------------------------
# SCOREBOARD (LIGA COMPLETA)
# -------------------------
def _mmddyyyy(d: datetime) -> str:
    return d.strftime("%m/%d/%Y")


@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_scoreboard_by_date(game_date_mmddyyyy: str):
    sb = scoreboardv2.ScoreboardV2(game_date=game_date_mmddyyyy)
    dfs = sb.get_data_frames()
    game_header = dfs[0].copy() if len(dfs) > 0 else pd.DataFrame()
    line_score  = dfs[1].copy() if len(dfs) > 1 else pd.DataFrame()
    return game_header, line_score


def build_league_games_for_date(day: datetime) -> pd.DataFrame:
    """
    Una fila por partido (liga completa) para una fecha.
    Incluye GAME_ID y TEAM_IDs para poder linkear al detalle.
    Robusto: NO depende de LineScore['MATCHUP'].
    """
    gh, ls = fetch_scoreboard_by_date(_mmddyyyy(day))
    if gh.empty:
        return pd.DataFrame(columns=[
            "GAME_ID", "Fecha", "Estado/Hora",
            "AWAY_ID", "HOME_ID",
            "Away", "Home", "Marcador"
        ])

    if ls is None or ls.empty:
        ls = pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"])

    id2abbr, id2name = team_maps()

    rows = []
    for _, g in gh.iterrows():
        game_id = g.get("GAME_ID", None)
        home_id = g.get("HOME_TEAM_ID", None)
        away_id = g.get("VISITOR_TEAM_ID", None)

        if pd.isna(game_id) or pd.isna(home_id) or pd.isna(away_id):
            continue

        g_ls = ls[ls["GAME_ID"] == game_id].copy() if "GAME_ID" in ls.columns else pd.DataFrame()

        def _team_pts(team_id):
            if not g_ls.empty and "TEAM_ID" in g_ls.columns and "PTS" in g_ls.columns:
                row = g_ls[g_ls["TEAM_ID"] == team_id]
                if not row.empty:
                    return row.iloc[0].get("PTS", None)
            return None

        home_name = id2name.get(int(home_id), id2abbr.get(int(home_id), ""))
        away_name = id2name.get(int(away_id), id2abbr.get(int(away_id), ""))

        home_pts = _team_pts(home_id)
        away_pts = _team_pts(away_id)

        status_text = str(g.get("GAME_STATUS_TEXT", "")).strip()

        marcador = ""
        if pd.notna(away_pts) and pd.notna(home_pts):
            try:
                marcador = f"{int(away_pts)} - {int(home_pts)}"
            except Exception:
                marcador = f"{away_pts} - {home_pts}"

        rows.append({
            "GAME_ID": str(game_id),
            "Fecha": day.date(),
            "Estado/Hora": status_text,
            "AWAY_ID": int(away_id),
            "HOME_ID": int(home_id),
            "Away": away_name,
            "Home": home_name,
            "Marcador": marcador
        })

    return pd.DataFrame(rows)


def build_recent_league_results(days_back: int) -> pd.DataFrame:
    """Junta resultados de los últimos N días (hoy-1, hoy-2, etc)."""
    anchor = datetime.today()
    all_rows = []
    for i in range(1, days_back + 1):
        day = anchor - timedelta(days=i)
        dfd = build_league_games_for_date(day)
        if not dfd.empty:
            all_rows.append(dfd)

    if not all_rows:
        return pd.DataFrame(columns=[
            "GAME_ID", "Fecha", "Estado/Hora",
            "AWAY_ID", "HOME_ID",
            "Away", "Home", "Marcador"
        ])

    out = pd.concat(all_rows, ignore_index=True)
    return out.sort_values(["Fecha"], ascending=False).reset_index(drop=True)


def to_day_agenda(df_games: pd.DataFrame) -> pd.DataFrame:
    """Fecha | Juegos (texto) agenda."""
    if df_games.empty:
        return df_games

    tmp = df_games.copy()

    def fmt(r):
        if str(r.get("Marcador", "")).strip():
            return f"{r['Away']} @ {r['Home']} — {r['Marcador']}"
        return f"{r['Away']} @ {r['Home']} — {r.get('Estado/Hora','')}"

    tmp["Juego"] = tmp.apply(fmt, axis=1)
    cal = tmp.groupby("Fecha")["Juego"].apply(lambda x: "  \n".join(list(x))).reset_index()
    cal = cal.rename(columns={"Juego": "Juegos"})
    return cal.sort_values("Fecha", ascending=False).reset_index(drop=True)


# -------------------------
# TEAM LEADERS (por equipo / temporada)
# -------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_team_player_stats(team_id: int, season: str, season_type: str) -> pd.DataFrame:
    """
    Trae stats de jugadores del equipo para la temporada.
    Una llamada por equipo.
    """
    ld = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=season_type,
        team_id_nullable=team_id
    )
    df = ld.get_data_frames()[0].copy()
    return df


def compute_team_leaders(team_id: int, season: str, season_type: str) -> pd.DataFrame:
    df = fetch_team_player_stats(team_id, season, season_type)
    if df.empty:
        return pd.DataFrame(columns=["Métrica", "Jugador", "Valor"])

    needed = ["PLAYER_NAME", "PTS", "REB", "AST", "FG3M", "MIN"]
    for c in needed:
        if c not in df.columns:
            df[c] = "" if c == "PLAYER_NAME" else 0

    metrics = [
        ("PTS",  "Puntos (PTS)"),
        ("REB",  "Rebotes (REB)"),
        ("AST",  "Asistencias (AST)"),
        ("FG3M", "Triples (3PM)"),
        ("MIN",  "Minutos (MIN)")
    ]

    rows = []
    for col, label in metrics:
        sub = df[["PLAYER_NAME", col]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0)
        best = sub.sort_values(col, ascending=False).head(1)
        if best.empty:
            continue
        rows.append({
            "Métrica": label,
            "Jugador": str(best.iloc[0]["PLAYER_NAME"]),
            "Valor": float(best.iloc[0][col])
        })

    return pd.DataFrame(rows)


# -------------------------
# CHART: barras + promedio + eje X en 2 filas
# -------------------------
def bar_with_avg(df: pd.DataFrame, col: str, title: str):
    avg = float(df[col].mean()) if len(df) else 0.0

    chart_df = df[["GameOrder", "FechaLbl", "OppLbl", col]].copy()
    chart_df["Promedio"] = avg
    chart_df["AboveAvg"] = chart_df[col] >= avg

    GREEN = "#22C55E"
    RED   = "#EF4444"
    AVG   = "#F9F9F9"

    bars = alt.Chart(chart_df).mark_bar(
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4
    ).encode(
        x=alt.X(
            "GameOrder:O",
            title=None,
            sort="ascending",
            axis=alt.Axis(labels=False, ticks=False, domain=False)
        ),
        y=alt.Y(f"{col}:Q", title=title),
        color=alt.condition(
            alt.datum.AboveAvg,
            alt.value(GREEN),
            alt.value(RED)
        )
    )

    avg_line = alt.Chart(chart_df).mark_rule(
        strokeDash=[6, 6],
        strokeWidth=2,
        color=AVG
    ).encode(y="Promedio:Q")

    label_df = pd.DataFrame({
        "GameOrder": [int(chart_df["GameOrder"].max())],
        "Promedio": [avg],
        "Label": [f"Avg {avg:.1f}"]
    })

    avg_label = alt.Chart(label_df).mark_text(
        align="left",
        baseline="bottom",
        dx=8,
        dy=-4,
        fontSize=12,
        color=AVG
    ).encode(
        x="GameOrder:O",
        y="Promedio:Q",
        text="Label:N"
    )

    main = (bars + avg_line + avg_label).properties(height=230)

    xlabels_date = alt.Chart(chart_df).mark_text(
        dy=10,
        fontSize=11,
        color="#C9D1D9"
    ).encode(
        x=alt.X("GameOrder:O", sort="ascending", axis=None),
        y=alt.value(0),
        text="FechaLbl:N"
    ).properties(height=18)

    xlabels_opp = alt.Chart(chart_df).mark_text(
        dy=10,
        fontSize=11,
        color="#C9D1D9"
    ).encode(
        x=alt.X("GameOrder:O", sort="ascending", axis=None),
        y=alt.value(0),
        text="OppLbl:N"
    ).properties(height=18)

    return alt.vconcat(main, xlabels_date, xlabels_opp, spacing=0)


# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("Configuración")

if st.sidebar.button("🔄 Actualizar datos"):
    st.cache_data.clear()
    st.rerun()

current_season = current_season_str()
season_options = [current_season, "2024-25", "2023-24", "2022-23", "2021-22", "2020-21"]
season = st.sidebar.selectbox("Temporada", season_options, index=0)

season_type = st.sidebar.selectbox("Tipo de temporada", ["Regular Season", "Playoffs"], index=0)
solo_local = st.sidebar.checkbox("Solo juegos de local", value=False)

teams_df = get_teams_df()
team_name = st.sidebar.selectbox("Equipo", teams_df["full_name"].tolist(), index=0)
team_id = int(teams_df.loc[teams_df["full_name"] == team_name, "id"].iloc[0])

try:
    roster_df = fetch_team_roster(team_id, season)
except Exception as e:
    st.error("No pude traer el roster del equipo.")
    st.code(str(e))
    st.stop()

if roster_df.empty:
    st.warning("Roster vacío para ese equipo/temporada. Prueba otra temporada.")
    st.stop()

player_name = st.sidebar.selectbox("Jugador (plantilla)", roster_df["PLAYER"].tolist(), index=0)
player_id = int(roster_df.loc[roster_df["PLAYER"] == player_name, "PLAYER_ID"].iloc[0])

try:
    df_all = fetch_gamelog(player_id, season, season_type)
except Exception as e:
    st.error("No pude traer juegos del jugador (la fuente a veces limita solicitudes).")
    st.code(str(e))
    st.stop()

if df_all.empty:
    st.warning("No hay juegos para ese jugador en esa temporada/tipo. Cambia Regular/Playoffs o temporada.")
    st.stop()

max_juegos = max(3, len(df_all))
n_ultimos = st.sidebar.slider("Últimos N juegos", min_value=3, max_value=max_juegos, value=min(10, max_juegos))


# -------------------------
# FILTRADO (estable) + labels eje X
# -------------------------
df = df_all.copy()
if solo_local:
    df = df[df["Local"] == True].copy()

df = df.tail(n_ultimos).reset_index(drop=True)
df["GameOrder"] = range(1, len(df) + 1)
df["FechaLbl"] = df["GAME_DATE"].dt.strftime("%m/%d")
df["OppLbl"] = df.apply(lambda r: f"{'vs' if r['Local'] else '@'} {r['Rival']}", axis=1)

last_date = df_all["GAME_DATE"].max().date()


# -------------------------
# LAYOUT
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧾 Tabla", "🧠 Insights", "📅 Calendario"])

metrics = [
    ("PTS", "PTS (Puntos)"),
    ("REB", "REB (Rebotes)"),
    ("AST", "AST (Asistencias)"),
    ("3PM", "3PM (Triples)"),
    ("PR", "PR (PTS+REB)"),
    ("PA", "PA (PTS+AST)"),
    ("PRA", "PRA (PTS+REB+AST)"),
    ("MIN", "MIN (Minutos)"),
]

with tab1:
    st.subheader(f"{team_name} — {player_name}")
    st.caption(f"{season} · {season_type} · Último juego registrado: {last_date}")

    st.markdown("### Promedios del rango seleccionado")
    cols = st.columns(4)
    for i, (col, label) in enumerate(metrics):
        cols[i % 4].metric(label, f"{df[col].mean():.1f}")

    st.divider()

    left, right = st.columns(2)
    for idx, (col, label) in enumerate(metrics):
        chart = bar_with_avg(df, col, label)

        if idx % 2 == 0:
            with left:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown(f"**{label}**")
                st.altair_chart(chart, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            with right:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown(f"**{label}**")
                st.altair_chart(chart, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("Detalle de juegos")
    df_show = df.copy()
    df_show["GAME_DATE"] = df_show["GAME_DATE"].dt.date
    df_show["Local"] = df_show["Local"].map(lambda x: "Local" if x else "Visita")

    cols_tbl = ["GAME_DATE", "Rival", "Local", "WL", "MIN", "PTS", "REB", "AST", "3PM", "PR", "PA", "PRA"]
    st.dataframe(df_show[cols_tbl], use_container_width=True)

with tab3:
    st.subheader("Insights rápidos (por métrica)")
    for col, label in metrics:
        avg = df[col].mean()
        std = df[col].std()

        st.markdown(f"### {label}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Promedio", f"{avg:.1f}")
        c2.metric("Máximo", f"{df[col].max():.1f}")
        c3.metric("Consistencia (STD)", f"{std:.2f}")

        st.divider()


# -------------------------
# CALENDARIO LIGA COMPLETA: últimos resultados + hoy + links a detalle
# -------------------------
with tab4:
    st.subheader("Calendario NBA — Liga completa")
    st.caption("Resultados recientes y partidos de HOY. Cada juego tiene link a detalle (líderes por equipo).")

    # ---------- Query param handling (robusto) ----------
    # Streamlit moderno: st.query_params (obj dict-like)
    selected_game_id = None
    try:
        if "game_id" in st.query_params:
            selected_game_id = st.query_params["game_id"]
    except Exception:
        selected_game_id = None

    if isinstance(selected_game_id, list):
        selected_game_id = selected_game_id[0] if selected_game_id else None

    # ---------- Detalle ----------
    if selected_game_id:
        st.markdown("### 🔎 Detalle del partido")
        st.markdown(
            f'<div class="mini-muted">Game ID: <b>{selected_game_id}</b> · '
            f'<a href="?">Volver al calendario</a></div>',
            unsafe_allow_html=True
        )

        hoy = datetime.today()
        df_today_all = build_league_games_for_date(hoy)
        df_recent_all = build_recent_league_results(10)

        df_pool = pd.concat([df_today_all, df_recent_all], ignore_index=True) if not df_recent_all.empty else df_today_all
        game_row = df_pool[df_pool["GAME_ID"].astype(str) == str(selected_game_id)]

        if game_row.empty:
            st.warning("No pude ubicar ese partido en HOY/recientes. Prueba abrirlo desde la tabla.")
        else:
            g = game_row.iloc[0]
            away_id = int(g["AWAY_ID"])
            home_id = int(g["HOME_ID"])
            away_name = str(g["Away"])
            home_name = str(g["Home"])

            st.markdown(f"**{away_name} @ {home_name}**")
            if str(g.get("Marcador", "")).strip():
                st.write(f"Marcador: {g['Marcador']}")
            if str(g.get("Estado/Hora", "")).strip():
                st.write(f"Estado/Hora: {g['Estado/Hora']}")

            st.divider()

            cA, cB = st.columns(2)

            with cA:
                st.markdown(f"### {away_name}")
                try:
                    leaders_away = compute_team_leaders(away_id, season, season_type)
                    st.dataframe(leaders_away, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error("No pude calcular líderes para este equipo (rate limit / endpoint).")
                    st.code(str(e))

            with cB:
                st.markdown(f"### {home_name}")
                try:
                    leaders_home = compute_team_leaders(home_id, season, season_type)
                    st.dataframe(leaders_home, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error("No pude calcular líderes para este equipo (rate limit / endpoint).")
                    st.code(str(e))

        st.divider()
        st.caption("Tip: si quieres más métricas (STL, BLK, TOV, FG%, FT%), dímelas y las agrego.")
        st.stop()  # evita renderizar el calendario abajo mientras estás en detalle

    # ---------- Calendario ----------
    top1, top2 = st.columns([1, 1])

    with top1:
        st.markdown('<div class="cal-card">', unsafe_allow_html=True)
        st.markdown("### 🧾 Últimos resultados (liga completa)")
        dias_back = st.slider("Días hacia atrás (resultados)", 1, 10, 3, key="dias_back_league")

        df_recent = build_recent_league_results(dias_back)

        if df_recent.empty:
            st.info("No pude traer resultados recientes (a veces el endpoint no devuelve data).")
        else:
            df_recent_view = df_recent.copy()
            # IMPORTANTE: link relativo (NO "/") para que no te mande a la home del dominio
            df_recent_view["Ver"] = df_recent_view["GAME_ID"].apply(lambda gid: f"?game_id={gid}")

            show_cols = ["Fecha", "Estado/Hora", "Away", "Home", "Marcador", "Ver"]
            st.dataframe(
                df_recent_view[show_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ver": st.column_config.LinkColumn("Ver", display_text="Abrir")
                }
            )

            st.markdown("**Vista por jornada (agenda):**")
            st.dataframe(to_day_agenda(df_recent), use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with top2:
        st.markdown('<div class="cal-card">', unsafe_allow_html=True)
        st.markdown("### 📅 HOY: partidos (liga completa)")

        hoy = datetime.today()
        df_today = build_league_games_for_date(hoy)

        if df_today.empty:
            st.info("Hoy no aparecen juegos (o el endpoint no devolvió data).")
        else:
            df_today_view = df_today.copy()
            df_today_view["TieneMarcador"] = df_today_view["Marcador"].astype(str).str.len() > 0
            df_today_view = df_today_view.sort_values("TieneMarcador", ascending=True).drop(columns=["TieneMarcador"])

            # IMPORTANTE: link relativo (NO "/")
            df_today_view["Ver"] = df_today_view["GAME_ID"].apply(lambda gid: f"?game_id={gid}")

            show_cols = ["Fecha", "Estado/Hora", "Away", "Home", "Marcador", "Ver"]
            st.dataframe(
                df_today_view[show_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ver": st.column_config.LinkColumn("Ver", display_text="Abrir")
                }
            )

            st.markdown("**Vista tipo agenda:**")
            st.dataframe(to_day_agenda(df_today), use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.caption("Siguiente mejora: calendario semanal en cuadritos + filtro por día + buscar equipo.")

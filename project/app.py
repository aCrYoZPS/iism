"""Interactive walk-through of the project for the presentation.

    streamlit run app.py

Nine screens in the order of results/story.md. Screens 2 and 5 simulate live: the chosen game is
fitted and 200 trajectories are drawn on the spot; everything "population-side" (template, plateau
rule, neighbours, stretch factor) is taken from the other folds, exactly as in the scripts.
Reads processed/*.csv, results/*.txt and the Anton games.csv (titles). Nothing is written.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import check_fits
import fit_decay
import forecast_curves as fc
import forecast_improve as fi

DATA_DIR = Path("datasets/kaggle/antonkozyriev-game-recommendations-on-steam")
PROCESSED, RESULTS = Path("processed"), Path("results")
WINDOW, N_SIMS = 104, 200

# fixed colour per entity (validated default palette of the dataviz skill, light mode)
COLOR = {
    "reality": "#0b0b0b", "flat": "#52514e", "exp": "#eb6834", "power": "#2a78d6",
    "lognormal": "#1baf7a", "hawkes50": "#6250d6", "shock_rule": "#e34948", "template": "#eda100",
    "power_c_pop": "#008300", "power_c_shrink": "#008300", "power_c_reg": "#e87ba4",
    "reg_conf": "#e87ba4", "template_knn": "#eda100", "mix_shrink_knn": "#e34948",
    "mix_shrink_template": "#e34948", "own": "#2a78d6", "control": "#52514e",
    "free": "#2a78d6", "shrink": "#eb6834", "pop": "#1baf7a",
    "reg_shock": "#6250d6", "mix_shock_knn": "#6250d6", "reg_ext": "#1baf7a",
}
LABEL = {
    "reality": "реальность", "flat": "плоская планка", "exp": "экспонента", "power": "степенной закон",
    "lognormal": "логнормальная форма", "hawkes50": "Хокс (η ≤ 0,5)", "shock_rule": "толчки",
    "template": "шаблон (другие игры)", "power_c_pop": "степенной + плато из популяции",
    "power_c_shrink": "степенной + плато со сжатием", "power_c_reg": "финальная: степенной + плато из регрессии",
    "reg_conf": "финальная, калиброванная", "template_knn": "шаблон по 50 соседям",
    "mix_shrink_knn": "смесь: степенной с плато + соседи", "mix_shrink_template": "смесь: степенной с плато + шаблон",
    "free": "свободное C", "shrink": "C со сжатием", "pop": "C из популяции",
    "reg_shock": "финальная + толчки", "mix_shock_knn": "смесь: финальная с толчками + соседи",
    "reg_ext": "финальная, плато из расширенной регрессии",
}
STAT_LABEL = {"total": "общее число отзывов", "peak_week": "неделя пика",
              "early_share": "доля отзывов в первые 8 недель", "t50": "неделя 50% отзывов"}
REF_LINE = "#8a8984"  # reference lines: mid grey, visible on light and dark backgrounds
BUCKET_LABEL = {"12-25": "недели 12–25", "26-51": "недели 26–51", "52-103": "недели 52–103", "total": "сумма"}

st.set_page_config(page_title="Жизненный цикл игры в Steam", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.2rem; max-width: 1400px;}
h1 {font-size: 2.0rem !important;} h2 {font-size: 1.5rem !important;}
.lead {font-size: 1.15rem; color: #52514e; line-height: 1.5;}
[data-testid="stMetricValue"] {font-size: 2.2rem;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data ------------------------------------
@st.cache_data(show_spinner=False)
def load_core() -> dict:
    series = pd.read_csv(PROCESSED / "weekly_series.csv")
    groups = pd.read_csv(PROCESSED / "games_groups.csv", parse_dates=["first_review", "date_release"])
    titles = pd.read_csv(DATA_DIR / "games.csv", usecols=["app_id", "title"]).set_index("app_id")["title"]
    simple = pd.read_csv(PROCESSED / "fits_simple.csv")
    hawkes = pd.read_csv(PROCESSED / "fits_hawkes.csv")
    shocks = pd.read_csv(PROCESSED / "fits_shocks.csv")
    fits = simple.merge(hawkes.drop(columns=["group"]), on="app_id", how="left") \
                 .merge(shocks.drop(columns=["group"]), on="app_id", how="left").set_index("app_id")
    keep = simple[(simple["group"] == "main") & (simple["n_weeks"] == WINDOW)]
    ids = np.sort(keep["app_id"].to_numpy())
    Y = series[series["app_id"].isin(ids)].pivot(index="app_id", columns="week", values="n_reviews") \
              .loc[ids].to_numpy(dtype=float)
    totals = pd.Series(Y.sum(axis=1), index=ids)
    names = {int(a): f"{titles.get(a, a)} ({a})" for a in ids}
    return dict(series=series, groups=groups, fits=fits, ids=ids, Y=Y, totals=totals, names=names,
                titles=titles)


@st.cache_resource(show_spinner="Считаю популяционные величины по фолдам (один раз)…")
def population(L: int) -> dict:
    core = load_core()
    Y, ids = core["Y"], core["ids"]
    folds = np.random.default_rng(fc.SEED).permutation(len(ids)) % fc.N_FOLDS
    taus, rs, kappas, sigmas = fc.build_templates(Y, folds, L)
    tags = fi.load_games(DATA_DIR).reindex(ids)["first_tag"].fillna("none").to_numpy()
    kappa_reg, sigma_reg, slope = fi.plateau_regression(Y, folds, L, tags)
    tau_knn = fi.knn_templates(Y, folds, L)
    return dict(folds=folds, taus=taus, rs=rs, kappas=kappas, sigmas=sigmas, kappa_reg=kappa_reg,
                tau_knn=tau_knn, slope=slope)


@st.cache_data(show_spinner=False)
def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / name)


def text_file(name: str) -> str:
    p = RESULTS / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


# ---------------------------------------------------------------- helpers ---------------------------------
def base_layout(fig: go.Figure, height: int = 460, **kw) -> go.Figure:
    hovermode = kw.pop("hovermode", "x unified")
    fig.update_layout(template="simple_white", height=height, font=dict(size=15),
                      margin=dict(l=50, r=20, t=50, b=50), legend=dict(orientation="h", y=-0.18),
                      hovermode=hovermode, **kw)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e8e7e3")
    return fig


def band(fig: go.Figure, t: np.ndarray, draws: np.ndarray, key: str, show_paths: int = 0, row=None, col=None):
    """5–95% band of the simulations, median line, optional sample paths."""
    lo, med, hi = np.percentile(draws, [5, 50, 95], axis=0)
    c = COLOR[key]
    rgba = f"rgba({int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},0.18)"
    args = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", fillcolor=rgba,
                             line=dict(width=0), hoverinfo="skip", name=f"{LABEL[key]}: 90% коридор",
                             showlegend=False), **args)
    fig.add_trace(go.Scatter(x=t, y=med, mode="lines", line=dict(color=c, width=2), name=LABEL[key]), **args)
    for k in range(show_paths):
        fig.add_trace(go.Scatter(x=t, y=draws[k], mode="lines", line=dict(color=c, width=1, dash="dot"),
                                 opacity=0.5, showlegend=False, hoverinfo="skip"), **args)


def reality(fig: go.Figure, t: np.ndarray, y: np.ndarray, row=None, col=None, name="реальность"):
    args = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color=COLOR["reality"], width=2.5), name=name), **args)


def game_picker(core: dict, key: str, default_rank: int = 0) -> int:
    order = core["totals"].sort_values(ascending=False).index
    options = [int(a) for a in order]
    return st.selectbox("Игра (можно набрать название)", options, index=default_rank,
                        format_func=lambda a: core["names"][a], key=key)


def lead(text: str) -> None:
    st.markdown(f'<p class="lead">{text}</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------- screens ---------------------------------
def screen_intro(core: dict) -> None:
    st.title("Жизненный цикл игры в Steam: имитация, сравнение моделей, прогноз")
    lead("Как живёт игра после выхода и что можно сказать по первым неделям? Онлайн новых игр не записан, "
         "а отзывы есть с первой недели у каждой, поэтому ряд отзывов — заменитель онлайна (это проверено, экран 8).")
    g = core["groups"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("отзывов в наборе", "41 млн", "2010–2022")
    c2.metric("игр с ≥ 200 отзывов", f"{(g['group'] != 'excluded').sum():,}".replace(",", " "))
    c3.metric("основная группа", f"{(g['group'] == 'main').sum():,}".replace(",", " "), "старт жизни наблюдается")
    c4.metric("с полным окном 104 недели", f"{len(core['ids']):,}".replace(",", " "))
    col1, col2 = st.columns([1.1, 1])
    with col1:
        reasons = g.groupby(["group", "reason"]).size().reset_index(name="n").sort_values("n", ascending=True)
        reasons["label"] = reasons["reason"].str.replace("released before review record starts", "вышли до начала записи") \
            .str.replace("fewer than 200 reviews", "меньше 200 отзывов") \
            .str.replace("first review before release", "первый отзыв раньше релиза (ранний доступ)") \
            .str.replace("first review more than 30 days after release", "первый отзыв позже 30 дней после релиза") \
            .str.replace("release date missing", "нет даты релиза")             .str.replace("first review within window of release", "основная группа: старт виден")             .str.replace("first review > 30 days after release", "первый отзыв позже 30 дней после релиза")
        fig = go.Figure(go.Bar(x=reasons["n"], y=reasons["label"], orientation="h",
                               marker=dict(color=[COLOR["power"] if gr == "main" else COLOR["flat"] for gr in reasons["group"]]),
                               text=reasons["n"].map(lambda v: f"{v:,}".replace(",", " ")), textposition="outside"))
        base_layout(fig, 380, title="Куда делись игры: отбор", hovermode="y")
        fig.update_xaxes(title="игр", showticklabels=False)
        st.plotly_chart(fig, width='stretch')
    with col2:
        Y = core["Y"]
        share = Y / np.clip(Y.sum(axis=1, keepdims=True), 1, None)
        t = np.arange(WINDOW)
        lo, med, hi = np.percentile(share, [25, 50, 75], axis=0)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                                 fillcolor="rgba(42,120,214,0.18)", line=dict(width=0), name="квартили"))
        fig.add_trace(go.Scatter(x=t, y=med, line=dict(color=COLOR["power"], width=2), name="медиана по играм"))
        base_layout(fig, 380, title="Типичная форма: доля отзывов по неделям", yaxis_type="log")
        fig.update_xaxes(title="неделя от первого отзыва")
        st.plotly_chart(fig, width='stretch')
    st.caption("Неделя 0 = неделя первого отзыва; внутри недели отзывы суммируются. У 69% игр пик в первые 8 недель, "
               "у 23% — позже 26-й недели.")


def screen_simulation(core: dict) -> None:
    st.title("Модель = генератор. Сравниваем по тому, что она порождает")
    lead("Каждая модель задаёт среднюю линию и случайность вокруг неё (отрицательное биномиальное распределение). "
         "Из подогнанной модели порождаем 200 рядов и смотрим, похожи ли они на реальный. "
         "Хокс лучше по правдоподобию, но его коридоры в свободной имитации расползаются.")
    c1, c2 = st.columns([2, 1])
    with c1:
        app_id = game_picker(core, "sim_game", 5)
    with c2:
        models = st.multiselect("Модели", ["exp", "power", "lognormal", "hawkes50", "shock_rule"],
                                default=["power", "hawkes50"], format_func=lambda m: LABEL[m], key="sim_models")
    paths = st.slider("Показать отдельные имитации", 0, 5, 2, key="sim_paths")
    i = int(np.searchsorted(core["ids"], app_id))
    y = core["Y"][i]
    t = np.arange(WINDOW)
    row = core["fits"].loc[app_id]
    rng = np.random.default_rng(app_id)
    fig = go.Figure()
    stats_rows = []
    obs = {k: v[0] for k, v in check_fits.statistics(y).items()}
    for m in models:
        if not bool(row.get(f"{m}_converged", False)):
            st.warning(f"{LABEL[m]}: подгонка не сошлась для этой игры")
            continue
        if m in check_fits.RECURSIVE_NAMES:
            draws = check_fits.simulate_hawkes(row, m, WINDOW, N_SIMS, rng)
        else:
            draws = check_fits.simulate(check_fits.mean_curve(m, row, t.astype(float)), row[f"{m}_r"], N_SIMS, rng)
        band(fig, t, draws, m, paths)
        sims = check_fits.statistics(draws)
        for s in check_fits.STAT_NAMES:
            stats_rows.append(dict(модель=LABEL[m], характеристика=STAT_LABEL[s], факт=round(obs[s], 2),
                                   медиана_имитаций=round(float(np.median(sims[s])), 2),
                                   коридор_90=f"{np.percentile(sims[s], 5):.1f} … {np.percentile(sims[s], 95):.1f}",
                                   CRPS=round(check_fits.crps_for(s, obs[s], sims[s]), 3)))
    reality(fig, t, y)
    base_layout(fig, 480, title=core["names"][app_id], yaxis_type="log")
    fig.update_xaxes(title="неделя от первого отзыва")
    fig.update_yaxes(title="отзывов в неделю (лог)")
    st.plotly_chart(fig, width='stretch')
    if stats_rows:
        st.dataframe(pd.DataFrame(stats_rows), hide_index=True, width='stretch')
    st.caption("CRPS — расстояние между распределением имитаций и фактом (меньше — лучше). "
               "Коридор 90% должен накрывать факт примерно в 90% игр: если чаще — модель слишком широкая.")


def screen_no_winner(core: dict) -> None:
    st.title("Единой лучшей модели нет")
    lead("Восемь моделей, 200 имитаций на игру, четыре характеристики траектории. "
         "По доле на старте лучше Хокс, по неделе пика — логнормальная, по общему числу — экспонента. "
         "Правдоподобие и имитация дают разные рейтинги.")
    sc = csv("sim_check.csv")
    sc = sc[sc["group"] == "main"]
    order = ["exp", "power", "lognormal", "hawkes", "hawkes80", "hawkes50", "shock_rule", "shock_joint"]
    names = {"hawkes": "Хокс (η ≤ 0,99)", "hawkes80": "Хокс (η ≤ 0,8)", "shock_joint": "толчки (скрытые)"}
    names.update({k: LABEL[k] for k in ("exp", "power", "lognormal", "hawkes50", "shock_rule")})
    fig = make_subplots(rows=1, cols=4, subplot_titles=[STAT_LABEL[s] for s in check_fits.STAT_NAMES], shared_yaxes=False)
    for j, s in enumerate(check_fits.STAT_NAMES, start=1):
        part = sc[sc["stat"] == s].groupby("model")["crps"].mean().reindex(order)
        best = part.idxmin()
        fig.add_trace(go.Bar(x=[names[m] for m in order], y=part.values,
                             marker=dict(color=[COLOR["power"] if m == best else "#b9b8b2" for m in order]),
                             text=[f"{v:.3f}" for v in part.values], textposition="outside", showlegend=False), row=1, col=j)
    base_layout(fig, 430, title="Средний CRPS по характеристике (синим — лучшая модель для неё)", hovermode="closest")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, width='stretch')
    cov = sc.assign(inside=sc["percentile"].between(0.05, 0.95)).groupby(["model", "stat"])["inside"].mean().unstack("stat").reindex(order)
    cov.columns = [STAT_LABEL[c] for c in cov.columns]
    cov.index = [names[m] for m in cov.index]
    st.markdown("**Доля игр, у которых факт внутри 90%-коридора** (идеал 0,90; больше — коридор слишком широкий):")
    st.dataframe(cov.round(2), width='stretch')


def screen_types(core: dict) -> None:
    st.title("«Типы траекторий» данные не поддерживают")
    lead("Три описания игры (форма, сводные, параметры Хокса), два метода кластеризации, контроль на данных "
         "без структуры, проверка предсказуемости по первым 12 неделям без утечки. "
         "Устойчиво только грубое деление на две группы по доле отзывов на старте.")
    cc = csv("cluster_criteria.csv")
    lf = cc[(cc["leak_free"] == True) & (cc["method"] == "kmeans")]
    feat_names = {"shape_f": "форма (недели 12–103)", "summary_f": "сводные (12–103)", "params_f": "параметры Хокса (12–103)"}
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(feat_names.values()), shared_yaxes=True)
    for j, f in enumerate(feat_names, start=1):
        p = lf[lf["feature"] == f].sort_values("k")
        fig.add_trace(go.Scatter(x=p["k"], y=p["macro_f1"], mode="lines+markers", name="предсказание типа по неделям 0–11",
                                 line=dict(color=COLOR["power"], width=2), marker=dict(size=9), showlegend=j == 1), row=1, col=j)
        fig.add_trace(go.Scatter(x=p["k"], y=p["dummy_macro_f1"], mode="lines+markers", name="случайное угадывание",
                                 line=dict(color=COLOR["flat"], width=2, dash="dash"), marker=dict(size=9), showlegend=j == 1), row=1, col=j)
    base_layout(fig, 400, title="Насколько тип будущего предсказуем по первым 12 неделям (macro-F1, выше — лучше)", hovermode="x")
    fig.update_xaxes(title="число типов k")
    st.plotly_chart(fig, width='stretch')
    c1, c2, c3 = st.columns(3)
    c1.metric("силуэт формы будущего", "0,085", "шум без структуры: 0,082")
    c2.metric("игр с параметрами Хокса внутри границ", "30%", "остальные упираются в границу")
    c3.metric("связь типа с жанром (V Крамера)", "0,06–0,15", "с ценой и рейтингом ≈ 0")
    st.caption("Силуэт — мера разделённости кластеров; если он такой же, как у случайных данных с теми же "
               "корреляциями, кластеры ничего не значат.")


def screen_forecast(core: dict) -> None:
    st.title("Прогноз всей кривой по первым неделям: кривые уходят в ноль, игры — нет")
    lead("Обучаем по первым 12 (или 26) неделям, предсказываем остальные, сравниваем с фактом. "
         "Все кривые спада проигрывают даже плоской планке, потому что реальные игры выходят на плато. "
         "Добавление плато из популяции и смесь с «памятью» о похожих играх это чинят.")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        app_id = game_picker(core, "fc_game", 5)
    with c2:
        L = st.radio("Обучение по неделям", [12, 26], horizontal=True, key="fc_L")
    with c3:
        models = st.multiselect("Модели", ["flat", "exp", "power", "template", "power_c_pop", "power_c_reg", "mix_shrink_knn"],
                                default=["power", "template", "power_c_reg", "mix_shrink_knn"],
                                format_func=lambda m: LABEL[m], key="fc_models")
    pop = population(L)
    i = int(np.searchsorted(core["ids"], app_id))
    y, f = core["Y"][i], pop["folds"][i]
    rng = np.random.default_rng(app_id * 10 + L)
    sims, levels = fc.forecasts_for_game(y, L, pop["taus"][f], pop["rs"][f], pop["kappas"][f], pop["sigmas"][f], rng)
    S = max(float(y[:L].sum()), 1.0)
    t_train, t_future = np.arange(L, dtype=float), np.arange(L, WINDOW, dtype=float)
    smooth = np.convolve(y[:L], np.ones(4) / 4, mode="valid")
    reg = fc.fit_power_floor(y[:L].astype(float), t_train, max(float(smooth.max()), 1.0), 1.0, "pop", pop["kappa_reg"][i] * S, 1.0)
    sims["power_c_reg"] = fc.nb_draws(fc.power_floor_mean(reg["log_A"], reg["log_alpha"], reg["C"], t_future), reg["r"], rng)
    sims["template_knn"] = fc.nb_draws(pop["tau_knn"][i] * S, pop["rs"][f], rng)
    if "power_c_shrink" in sims:
        sims["mix_shrink_knn"] = np.vstack([sims["power_c_shrink"][:N_SIMS // 2], sims["template_knn"][:N_SIMS // 2]])
    fig = go.Figure()
    fig.add_vrect(x0=-0.5, x1=L - 0.5, fillcolor="#e8e7e3", opacity=0.5, line_width=0, annotation_text="обучение", annotation_position="top left")
    for m in models:
        if m in sims:
            band(fig, t_future, sims[m], m)
    reality(fig, np.arange(WINDOW), y)
    base_layout(fig, 500, title=core["names"][app_id], yaxis_type="log")
    fig.update_xaxes(title="неделя от первого отзыва")
    fig.update_yaxes(title="отзывов в неделю (лог)")
    st.plotly_chart(fig, width='stretch')
    rows = [dict(модель=LABEL[r["model"]], горизонт=BUCKET_LABEL[r["bucket"]], CRPS=round(r["crps"], 3),
                 покрытие_90=None if np.isnan(r["covered90"]) else round(r["covered90"], 2))
            for r in fc.score_game(y, L, {m: sims[m] for m in models if m in sims})]
    if rows:
        tab = pd.DataFrame(rows).pivot(index="модель", columns="горизонт", values="CRPS")
        st.markdown("**CRPS этой игры по горизонтам** (меньше — лучше):")
        st.dataframe(tab, width='stretch')
    st.caption(f"Плато из популяции для этой игры: {levels['c_pop']:.1f} отзывов/нед.; из регрессии по размеру и жанру: "
               f"{reg['C']:.1f}; реальный уровень в неделях 78–103: {fc.actual_plateau(y):.1f}. "
               "Всё «популяционное» взято с игр из других фолдов, эта игра в них не участвует.")


def screen_plateau(core: dict) -> None:
    st.title("Плато есть, но по коротким рядам его не оценить — показано имитацией")
    lead("Модель как генератор: 321 набор «истинных» параметров, по 20 искусственных игр, подгонка по 12 и 26 неделям. "
         "Свободное C у четверти игр уходит в ноль (ошибка в тысячи раз); сжатие к популяции убирает провал. "
         "Само плато растёт как S^0,55: крупные игры удерживают меньшую долю старта.")
    rp = csv("recover_params.csv")
    L = st.radio("Подгонка по неделям", [12, 26], horizontal=True, key="rp_L")
    part = rp[rp["L"] == L]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for mode in ("free", "shrink"):
            m = part[part["mode"] == mode]
            fig.add_trace(go.Histogram(x=np.log(np.clip(m["est_C"], 1e-3, None) / m["true_C"]), xbins=dict(start=-8, end=4, size=0.25),
                                       histnorm="probability density", name=LABEL[mode], marker=dict(color=COLOR[mode]), opacity=0.65))
        fig.add_vline(x=0, line=dict(color=REF_LINE, dash="dash", width=2))
        base_layout(fig, 420, title="Ошибка оценки плато: log(найденное C / истинное C)", barmode="overlay", hovermode="x")
        fig.update_xaxes(title="0 — точное попадание; −7 — C «схлопнулось» в ноль")
        st.plotly_chart(fig, width='stretch')
        at_bound = part[part["mode"] == "free"]["c_at_bound"].mean()
        st.metric("свободное C ушло в ноль", f"{at_bound:.0%} искусственных игр", "со сжатием: 0%")
    with c2:
        fig = go.Figure()
        for mode in ("free", "pop", "shrink"):
            m = part[part["mode"] == mode]
            fig.add_trace(go.Box(y=m["forecast_log_err"].clip(-4, 4), name=LABEL[mode], marker=dict(color=COLOR[mode]),
                                 boxpoints=False, line=dict(width=2)))
        fig.add_hline(y=0, line=dict(color=REF_LINE, dash="dash", width=2))
        base_layout(fig, 420, title=f"Ошибка прогноза суммы за недели {L}–103 (лог)", hovermode="closest", showlegend=False)
        st.plotly_chart(fig, width='stretch')
        err = part.groupby("mode")["forecast_log_err"].apply(lambda e: e.abs().mean())
        st.metric("средняя |ошибка| прогноза, сжатие", f"{err['shrink']:.3f}", f"свободное C: {err['free']:.3f}; из популяции: {err['pop']:.3f}", delta_color="off")
    ff = csv("forecast_floor_fits.csv")
    ff = ff[(ff["L"] == 12) & (ff["S"] > 0)]
    fig = go.Figure(go.Scattergl(x=ff["S"], y=ff["plateau_actual"], mode="markers", marker=dict(size=5, color=COLOR["power"], opacity=0.25),
                                 name="игра", hovertemplate="S=%{x}, плато=%{y:.1f}<extra></extra>"))
    xs = np.logspace(np.log10(ff["S"].min()), np.log10(ff["S"].max()), 50)
    b = np.polyfit(np.log(ff["S"]), np.log(ff["plateau_actual"]), 1)
    fig.add_trace(go.Scatter(x=xs, y=np.exp(b[1]) * xs ** b[0], mode="lines", line=dict(color=COLOR["mix_shrink_knn"], width=2.5),
                             name=f"степенная зависимость: плато ∝ S^{b[0]:.2f}"))
    base_layout(fig, 420, title="Реальное плато (недели 78–103) против стартового объёма S (отзывы за первые 12 недель)",
                xaxis_type="log", yaxis_type="log", hovermode="closest")
    fig.update_xaxes(title="S (лог)")
    fig.update_yaxes(title="отзывов в неделю на плато (лог)")
    st.plotly_chart(fig, width='stretch')


def _bootstrap_mean(x: np.ndarray, n: int = 500, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def screen_final(core: dict) -> None:
    st.title("Финальная модель и лучший достижимый прогноз")
    lead("Финальная (одна формула): y ~ NB(μ, r), μ = A(1+t)^(−α) + κ̂·S, где плато κ̂ берётся из регрессии "
         "log(плато/S) ~ log S + жанр по другим играм; коридоры растянуты конформным множителем, подобранным на чужих играх. "
         "Смесь 100 имитаций этой модели и 100 имитаций шаблона по 50 похожим играм — лучший прогноз из полученных.")
    sc = csv("forecast_improve_scores.csv")
    L = st.radio("Обучение по неделям", [12, 26], horizontal=True, key="fin_L")
    part = sc[sc["L"] == L]
    models = ["template", "power_c_shrink", "power_c_reg", "reg_conf", "mix_shrink_knn"]
    _crps_cov_bars(part, models, patterns={"reg_conf": "/"}, height=460)
    st.caption("Все модели проверены на отложенных неделях 5311 игр; шаблон, плато, соседи и множители считаются по четырём фолдам "
               "и применяются к пятому. Смесь — объединение 200 имитаций (смесь распределений, не среднее кривых), веса 50/50 заданы заранее.")


def screen_players(core: dict) -> None:
    st.title("Перенос на число игроков (Mendeley)")
    lead("Слева: отзывы повторяют форму онлайна — корреляция недельных изменений 0,42 против 0,09 у контроля "
         "(онлайн случайной другой игры на тех же календарных неделях). Справа: те же модели на онлайне дают то же ранжирование: "
         "плато нужно, сжатие лучше фиксированного, смесь лучше всех.")
    c1, c2 = st.columns(2)
    with c1:
        own, null = csv("proxy_scores.csv"), csv("proxy_null.csv")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=null["r_diff"], xbins=dict(start=-1, end=1, size=0.05), histnorm="probability density",
                                   name="контроль: онлайн чужой игры", marker=dict(color=COLOR["control"]), opacity=0.6))
        fig.add_trace(go.Histogram(x=own["r_diff"], xbins=dict(start=-1, end=1, size=0.05), histnorm="probability density",
                                   name="свой онлайн", marker=dict(color=COLOR["own"]), opacity=0.7))
        base_layout(fig, 400, title="Корреляция недельных изменений: отзывы × онлайн (338 игр)", barmode="overlay", hovermode="x")
        st.plotly_chart(fig, width='stretch')
        m1, m2, m3 = st.columns(3)
        m1.metric("медиана, свои пары", f"{own['r_diff'].median():.2f}", f"контроль {null['r_diff'].median():.2f}", delta_color="off")
        m2.metric("игр выше 95-го процентиля контроля", f"{own['above_control_diff'].mean():.0%}")
        m3.metric("лучший сдвиг = 0 недель", f"{(own['best_lag'] == 0).mean():.0%} игр", "отзывы не запаздывают")
    with c2:
        ms = csv("mendeley_forecast_scores.csv")
        part = st.radio("Постановка", ["record", "release"], horizontal=True, key="md_part",
                        format_func=lambda p: {"record": "1454 зрелые игры, от начала записи (обучение 26 недель)",
                                               "release": "62 игры после выхода из раннего доступа (обучение 12 недель)"}[p])
        tot = ms[(ms["part"] == part) & (ms["bucket"] == "total")].groupby("model")["crps"].mean()
        order = [m for m in ["exp", "lognormal", "flat", "template", "power", "power_c_pop", "power_c_shrink", "mix_shrink_template"] if m in tot]
        fig = go.Figure(go.Bar(x=tot[order].values, y=[LABEL[m] for m in order], orientation="h",
                               marker=dict(color=[COLOR[m] for m in order]), text=[f"{v:.3f}" for v in tot[order].values], textposition="outside"))
        base_layout(fig, 400, title="CRPS суммы за окно прогноза на онлайне (меньше — лучше)", hovermode="y", showlegend=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width='stretch')
        if part == "release":
            st.caption("Старый уровень до релиза известен, но модель с ним (`power_c_known`) не лучше плато из популяции: "
                       "после выхода из раннего доступа игра выходит на новый уровень (у 48% выше старого).")
        else:
            st.caption("Зрелые игры за два года теряют около 30% игроков; наивная планка здесь сильна, смесь обходит и её.")
    st.markdown("**Пример: отзывы и онлайн одной игры на общих неделях**")
    pick = own.sort_values("r_diff", ascending=False)
    opts = pick["app_id"].tolist()
    app_id = st.selectbox("Игра", opts, format_func=lambda a: f"{pick.set_index('app_id').loc[a, 'name']} (корреляция изменений {pick.set_index('app_id').loc[a, 'r_diff']:.2f})", key="px_game")
    g = core["groups"].set_index("app_id")
    first = g.loc[app_id, "first_review"]
    from mendeley_proxy import daily_means, weekly_players
    base = Path("datasets/mendeley/player_counts")
    file = next((p for part_dir in ("PlayerCountHistoryPart1", "PlayerCountHistoryPart2") for p in [base / part_dir / f"{app_id}.csv"] if p.exists()), None)
    yrev = core["series"][core["series"]["app_id"] == app_id].set_index("week")["n_reviews"].reindex(range(WINDOW)).to_numpy(dtype=float)
    players = weekly_players(daily_means(file), first) if file else np.full(WINDOW, np.nan)
    t = np.arange(WINDOW)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["отзывов в неделю", "средний онлайн за неделю"], vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=t, y=yrev, line=dict(color=COLOR["own"], width=2), name="отзывы"), row=1, col=1)
    ok = np.isfinite(players)
    fig.add_trace(go.Scatter(x=t[ok], y=players[ok], line=dict(color=COLOR["shrink"], width=2), name="онлайн"), row=2, col=1)
    base_layout(fig, 480, showlegend=False)
    fig.update_yaxes(type="log")
    fig.update_xaxes(title="неделя от первого отзыва", row=2, col=1)
    st.plotly_chart(fig, width='stretch')


def screen_conclusions(core: dict) -> None:
    st.title("Выводы")
    st.markdown("""
1. **Жизнь игры по отзывам — быстрый спад и плато.** Без плато любая кривая спада проигрывает наивному прогнозу.
2. **Плато по коротким рядам не оценивается** (показано имитацией); его надо брать из популяции, и оно растёт как S^0,55.
3. **Лучшая по правдоподобию модель не лучшая по имитации.** Хокс и толчки объясняют виденный ряд, но порождают слишком широкие коридоры.
4. **Типы траекторий данные не поддерживают**, кроме грубого деления на два.
5. **Прогноз по 12 неделям:** финальная формула лучше шаблона на всех горизонтах и откалибрована; смесь с «памятью» о похожих играх лучше ещё на четверть.
6. **Всё повторяется на числе игроков**, а отзывы доказанно повторяют форму онлайна.
7. **Толчки реальны, но редки (1% недель) и прогноз не улучшают:** промахи коридора — ошибки уровня, не скачки.
8. **На чужом наборе отзывов ранжирование и смесь переносятся,** а правило «плато ∝ S^0,55» зависит от охвата набора.
9. **Параметры говорят об играх:** бесплатные — резкий старт и низкое плато, дорогие — быстрый спад и высокое плато, положительные первые отзывы предвещают плато.
""")
    st.subheader("Ограничения")
    st.markdown("""
- Отбор по популярности (≥ 200 отзывов): выводы про игры с заметной аудиторией.
- Сравнение восьми моделей внутривыборочное; отложенная проверка есть только для прогноза.
- На онлайне отрицательное биномиальное распределение — рабочая модель разброса, коридоры там самоуверенны (0,3–0,76).
- Жанр — только первый тег; у 20% игр плато не находится даже по 104 неделям.
- Внешний набор — 178 популярных игр, и в нём вдвое больший охват отзывов: это проверка переноса, а не повторение на такой же выборке.
""")
    sens = text_file("sensitivity_summary.txt")
    if sens:
        with st.expander("Чувствительность к порогам, окну и разбиению на фолды"):
            st.code(sens)


def _crps_cov_bars(part: pd.DataFrame, models: list, patterns: dict | None = None, height: int = 420,
                   title_crps: str = "Средний CRPS по горизонтам (меньше — лучше; усы — 95% бутстреп)") -> None:
    patterns = patterns or {}
    buckets = [b for b in ["12-25", "26-51", "52-103", "total"] if b in set(part["bucket"])]
    fig = go.Figure()
    for m in models:
        means, los, his = [], [], []
        for b in buckets:
            x = part[(part["model"] == m) & (part["bucket"] == b)]["crps"].to_numpy()
            mean, lo, hi = _bootstrap_mean(x)
            means.append(mean); los.append(mean - lo); his.append(hi - mean)
        fig.add_trace(go.Bar(x=[BUCKET_LABEL[b] for b in buckets], y=means, name=LABEL[m],
                             marker=dict(color=COLOR[m], pattern_shape=patterns.get(m, "")),
                             error_y=dict(type="data", array=his, arrayminus=los, thickness=1.5),
                             text=[f"{v:.3f}" for v in means], textposition="outside"))
    base_layout(fig, height, title=title_crps, barmode="group", hovermode="x")
    st.plotly_chart(fig, width='stretch')
    cov = part[part["bucket"] != "total"].pivot_table(index="model", columns="bucket", values="covered90", aggfunc="mean").reindex(models)
    fig = go.Figure()
    for m in models:
        fig.add_trace(go.Bar(x=[BUCKET_LABEL[b] for b in cov.columns], y=cov.loc[m], name=LABEL[m],
                             marker=dict(color=COLOR[m], pattern_shape=patterns.get(m, "")),
                             text=[f"{v:.2f}" for v in cov.loc[m]], textposition="outside"))
    fig.add_hline(y=0.9, line=dict(color=REF_LINE, dash="dash", width=2), annotation_text="цель 0,90")
    base_layout(fig, 360, title="Покрытие 90%-коридора", barmode="group", hovermode="x", yaxis_range=[0, 1.08])
    st.plotly_chart(fig, width='stretch')


def screen_shocks(core: dict) -> None:
    st.title("Толчки в прогнозе почти ничего не меняют")
    lead("Толчок — неделя, где факт выше 99,5%-квантиля модели минимум на 5 отзывов: 1% недель, медианный скачок ×6,7, "
         "у трети игр их нет. В прогноз толчки добавлены с частотой и размером из чужих фолдов. "
         "Промахи коридора вверх лишь на 8–13% приходятся на недели-толчки: модель промахивается уровнем, а не редкими скачками.")
    sc = csv("forecast_shocks_scores.csv")
    L = st.radio("Обучение по неделям", [12, 26], horizontal=True, key="sh_L")
    part = sc[sc["L"] == L]
    models = ["power_c_reg", "reg_shock", "reg_conf", "mix_shrink_knn", "mix_shock_knn"]
    _crps_cov_bars(part, models, patterns={"reg_conf": "/", "mix_shock_knn": "/"})
    dg = csv("forecast_shocks_diag.csv")
    dg = dg[dg["L"] == L]
    rows = []
    for m in ["power_c_reg", "reg_conf", "reg_shock"]:
        rows.append({"модель": LABEL[m],
                     "промахов вверх, доля недель": round(dg[f"{m}_above"].sum() / dg["n_weeks"].sum(), 3),
                     "из них на неделях-толчках": round(dg[f"{m}_above_flag"].sum() / max(dg[f"{m}_above"].sum(), 1), 3),
                     "промахов вниз, доля недель": round(dg[f"{m}_below"].sum() / dg["n_weeks"].sum(), 3),
                     "CRPS на неделях-толчках": round(dg[f"{m}_crps_flag"].sum() / dg["n_flag"].sum(), 3),
                     "CRPS на остальных": round(dg[f"{m}_crps_noflag"].sum() / (dg["n_weeks"] - dg["n_flag"]).sum(), 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption("Недели-толчки размечены по полному ряду самой игры (это описание, не прогноз). "
               "Толчок с вероятностью 1% не попадает в 90%-коридор по построению, поэтому слепое конформное растяжение остаётся честнее.")


def screen_external(core: dict) -> None:
    st.title("Внешняя проверка: другой набор отзывов")
    lead("Набор najzeko: 178 популярных игр с полным окном, 166 из них есть и в Anton. Anton держит около половины их отзывов "
         "(медиана 0,51 во всех возрастах игры), форма та же (ρ = 0,95). Все популяционные величины взяты из Anton без этих игр. "
         "Финальная модель на чужом наборе недооценивает плато; после прореживания до половины (как в Anton) результат почти совпадает с теми же играми внутри Anton.")
    L = st.radio("Обучение по неделям", [12, 26], horizontal=True, key="ex_L")
    models = ["template_knn", "power_c_reg", "reg_conf", "mix_shrink_knn"]
    nz = csv("najzeko_scores.csv"); nz = nz[nz["L"] == L]
    th = csv("najzeko_scores_thin.csv"); th = th[th["L"] == L]
    games = csv("najzeko_games.csv")
    common = set(games[(games["group"] == "main") & (games["anton_group"] == "main")]["app_id"])
    an = csv("forecast_improve_scores.csv"); an = an[(an["L"] == L) & an["app_id"].isin(common)]
    variants = [("najzeko, как есть", nz), ("najzeko, прорежен до ½", th), ("те же игры внутри Anton", an)]
    buckets = [b for b in ["12-25", "26-51", "52-103", "total"] if b in set(nz["bucket"])]
    for metric, title, yr in (("crps", "Средний CRPS (меньше — лучше)", None), ("covered90", "Покрытие 90%-коридора", [0, 1.08])):
        fig = make_subplots(rows=1, cols=3, subplot_titles=[v[0] for v in variants], shared_yaxes=True)
        for j, (_, part) in enumerate(variants, start=1):
            bk = [b for b in buckets if metric == "crps" or b != "total"]
            for m in models:
                vals = [part[(part["model"] == m) & (part["bucket"] == b)][metric].mean() for b in bk]
                fig.add_trace(go.Bar(x=[BUCKET_LABEL[b] for b in bk], y=vals, name=LABEL[m],
                                     marker=dict(color=COLOR[m], pattern_shape="/" if m == "reg_conf" else ""),
                                     text=[f"{v:.2f}" for v in vals], textposition="outside", showlegend=j == 1), row=1, col=j)
            if metric == "covered90":
                labels = [BUCKET_LABEL[b] for b in bk]
                fig.add_trace(go.Scatter(x=labels, y=[0.9] * len(labels), mode="lines", name="цель 0,90",
                                         line=dict(color=REF_LINE, dash="dash", width=2), showlegend=j == 1,
                                         hoverinfo="skip"), row=1, col=j)
        base_layout(fig, 400, title=title, barmode="group", hovermode="x")
        if yr:
            fig.update_yaxes(range=yr)
        st.plotly_chart(fig, width='stretch')
    agree = csv("najzeko_agreement.csv")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["отношение сумм Anton / najzeko по игре", "корреляция формы (Спирмен по неделям)"])
    fig.add_trace(go.Histogram(x=agree["ratio"], nbinsx=30, marker=dict(color=COLOR["power"]), showlegend=False), row=1, col=1)
    fig.add_trace(go.Histogram(x=agree["rho_level"], nbinsx=30, marker=dict(color=COLOR["power"]), showlegend=False), row=1, col=2)
    base_layout(fig, 300, title=f"Согласие двух наборов на {len(agree)} общих играх", hovermode="closest")
    st.plotly_chart(fig, width='stretch')
    st.caption("Вывод: ранжирование моделей и смесь переносятся на чужой набор; масштабная часть (плато ∝ S^0,55) зависит от доли отзывов, "
               "попавших в набор, и на наборе с вдвое большим охватом плато занижено. Популярные игры и внутри Anton даются хуже.")


def screen_findings(core: dict) -> None:
    st.title("Параметры модели как факты об играх")
    lead("У каждой из 5311 игр по полному окну есть скорость спада α, плато относительно стартовых 12 недель и доля первых 4 недель. "
         "Бесплатные игры: самое низкое плато (0,67 от типичного для их размера) и самый резкий старт; дорогие (≥ 15 $) спадают быстрее всех, но плато выше. "
         "Доля положительных отзывов в первые 12 недель — предвестник плато: верхняя четверть 1,25 против 0,76 у нижней (Спирмен +0,17).")
    groups = csv("game_findings_groups.csv")
    kind = st.radio("Разрез", ["price", "year", "pos_train quartile", "tag"], horizontal=True, key="gf_kind",
                    format_func=lambda k: {"price": "цена", "year": "год выхода", "pos_train quartile": "доля положительных (первые 12 недель)",
                                           "tag": "теги (игра с тегом против остальных)"}[k])
    part = groups[groups["kind"] == kind].iloc[::-1]
    panels = [("alpha", "скорость спада α"), ("plateau_adj", "плато, отн. типичного для размера"), ("early_share", "доля первых 4 недель")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[t for _, t in panels], shared_yaxes=True, horizontal_spacing=0.04)
    for j, (m, _) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(x=part[m], y=part["level"], mode="markers", marker=dict(color=COLOR["power"], size=11),
                                 error_x=dict(type="data", array=part[f"{m}_hi"] - part[m], arrayminus=part[m] - part[f"{m}_lo"], thickness=1.5),
                                 name="медиана, 95% бутстреп", showlegend=j == 1,
                                 customdata=part["n"], hovertemplate="%{y}: %{x:.3f} (n = %{customdata})"), row=1, col=j)
        if kind == "tag":
            fig.add_trace(go.Scatter(x=part[f"{m}_rest"], y=part["level"], mode="markers", name="остальные игры",
                                     marker=dict(color=COLOR["flat"], size=9, symbol="line-ns-open", line=dict(width=2)), showlegend=j == 1), row=1, col=j)
        ref = groups[groups["kind"] == "all"].iloc[0][m]
        fig.add_trace(go.Scatter(x=[ref, ref], y=[part["level"].iloc[0], part["level"].iloc[-1]], mode="lines",
                                 line=dict(color=REF_LINE, dash="dash", width=2), name="все игры", showlegend=j == 1,
                                 hovertemplate=f"все игры: {ref:.3f}<extra></extra>"), row=1, col=j)
    base_layout(fig, 420 if kind != "tag" else 560, title="Медианы по группам (серый пунктир — все игры)", hovermode="closest")
    st.plotly_chart(fig, width='stretch')
    st.markdown("**Эти признаки известны уже на 12-й неделе, поэтому их можно отдать регрессии плато.** "
                "Цена, бесплатность, год, доля положительных и её тренд, 15 тегов: разброс остатка log(плато/S) падает с 1,00 до 0,95, прогноз финальной модели улучшается немного, но надёжно.")
    rs = csv("game_findings_reg_scores.csv")
    L = st.radio("Обучение по неделям", [12, 26], horizontal=True, key="gf_L")
    part = rs[rs["L"] == L]
    buckets = [b for b in ["12-25", "26-51", "52-103", "total"] if b in set(part["bucket"])]
    fig = go.Figure()
    for variant, key in (("base", "power_c_reg"), ("extended", "reg_ext")):
        means, los, his = [], [], []
        for b in buckets:
            x = part[(part["variant"] == variant) & (part["bucket"] == b)]["crps"].to_numpy()
            mean, lo, hi = _bootstrap_mean(x)
            means.append(mean); los.append(mean - lo); his.append(hi - mean)
        fig.add_trace(go.Bar(x=[BUCKET_LABEL[b] for b in buckets], y=means, name=LABEL[key], marker=dict(color=COLOR[key]),
                             error_y=dict(type="data", array=his, arrayminus=los, thickness=1.5),
                             text=[f"{v:.3f}" for v in means], textposition="outside"))
    base_layout(fig, 400, title="CRPS финальной модели: плато из базовой и из расширенной регрессии", barmode="group", hovermode="x")
    st.plotly_chart(fig, width='stretch')
    st.caption("Всё описательно: группы отличаются по размеру и возрасту, причинных утверждений нет. Рейтинг из каталога 2022 года в регрессию не идёт — он знает будущее; "
               "используется только доля положительных среди отзывов обучающих недель.")


SCREENS = {
    "1. Вопрос и данные": screen_intro,
    "2. Модель как генератор": screen_simulation,
    "3. Единой лучшей нет": screen_no_winner,
    "4. Типы траекторий": screen_types,
    "5. Прогноз: кривые уходят в ноль": screen_forecast,
    "6. Плато и идентифицируемость": screen_plateau,
    "7. Финальная модель и смесь": screen_final,
    "8. Перенос на игроков": screen_players,
    "9. Толчки в прогнозе": screen_shocks,
    "10. Внешняя проверка": screen_external,
    "11. Параметры как факты": screen_findings,
    "12. Выводы": screen_conclusions,
}


def main() -> None:
    core = load_core()
    choice = st.sidebar.radio("Экраны", list(SCREENS), label_visibility="collapsed")
    st.sidebar.caption("Экраны 2 и 5 считают имитации и прогноз для выбранной игры на лету.")
    SCREENS[choice](core)


if __name__ == "__main__":
    main()

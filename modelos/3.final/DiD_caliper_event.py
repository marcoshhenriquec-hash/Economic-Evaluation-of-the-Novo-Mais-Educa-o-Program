# ======================================================
# modelo_principal_matching_did_caliper025.py
#
# Matching por trajetória do IDEB 2009-2015
# + caliper 0.25
# + DiD/event study
# + teste conjunto de pré-tendência
# + modelos delta 2015-2019
# ======================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ======================================================
# 1. Caminhos
# ======================================================

path_painel = r"C:\Users\suporte1\Desktop\lai\painel_ideb_pnme_2005_2019.csv"

out_dir = r"C:\Users\suporte1\Desktop\lai"

output_balanceamento = out_dir + r"\balanceamento_matching_IDEB_caliper025.csv"
output_resultados = out_dir + r"\resultados_matching_DiD_eventstudy_caliper025.csv"
output_medias = out_dir + r"\medias_matching_DiD_eventstudy_caliper025.csv"

caliper = 0.25

pre_years = [2009, 2011, 2013, 2015]
anos_event_study = [2009, 2011, 2013, 2015, 2017, 2019]
anos_event = [2009, 2011, 2013, 2017, 2019]  # 2015 é a base
pre_test = ["tratado_x_2009", "tratado_x_2011", "tratado_x_2013"]


# ======================================================
# 2. Funções auxiliares
# ======================================================

def smd(x_t, x_c):
    """
    Standardized Mean Difference.
    """
    x_t = pd.Series(x_t).dropna()
    x_c = pd.Series(x_c).dropna()

    pooled_sd = np.sqrt((x_t.var(ddof=1) + x_c.var(ddof=1)) / 2)

    if pooled_sd == 0 or np.isnan(pooled_sd):
        return np.nan

    return (x_t.mean() - x_c.mean()) / pooled_sd


def resumo_balanceamento(df, etapa, momento, variaveis):
    linhas = []

    tratados = df[df["tratamento_2017"] == 1]
    controles = df[df["tratamento_2017"] == 0]

    for var in variaveis:
        if var not in df.columns:
            continue

        linhas.append({
            "etapa": etapa,
            "momento": momento,
            "variavel": var,
            "media_tratadas": tratados[var].mean(),
            "media_controles": controles[var].mean(),
            "dif_abs": tratados[var].mean() - controles[var].mean(),
            "smd": smd(tratados[var], controles[var])
        })

    return linhas


def two_way_demean(data, cols, entity_col="CO_ENTIDADE", time_col="ANO", max_iter=50, tol=1e-10):
    """
    Residualiza variáveis por efeitos fixos de escola e ano.
    Funciona para painel desbalanceado.
    """
    out = data[cols].astype(float).copy()
    prev = out.copy()

    for it in range(max_iter):
        out = out - out.groupby(data[entity_col]).transform("mean")
        out = out - out.groupby(data[time_col]).transform("mean")
        out = out + out.mean()

        diff = (out - prev).abs().max().max()

        if diff < tol:
            print(f"Demeaning convergiu em {it + 1} iterações.")
            break

        prev = out.copy()

    return out


def resumo_modelo(model, nome_modelo, etapa, outcome, n_obs, n_escolas):
    linhas = []

    for var in model.params.index:
        linhas.append({
            "modelo": nome_modelo,
            "etapa": etapa,
            "outcome": outcome,
            "variavel": var,
            "coef": model.params.get(var, np.nan),
            "erro_padrao": model.bse.get(var, np.nan),
            "t": model.tvalues.get(var, np.nan),
            "pvalor": model.pvalues.get(var, np.nan),
            "n_obs": n_obs,
            "n_escolas": n_escolas,
            "r2": getattr(model, "rsquared", np.nan),
            "wald_stat_pretrend": np.nan,
            "wald_pvalor_pretrend": np.nan,
            "wald_df": np.nan
        })

    return linhas


# ======================================================
# 3. Matching
# ======================================================

def preparar_base_wide_para_matching(painel, etapa, outcome):
    """
    Transforma o painel em base wide por escola para fazer matching.
    """

    d = painel.copy()
    d = d.dropna(subset=[outcome, "tratamento_2017"]).copy()

    wide = (
        d.pivot_table(
            index="CO_ENTIDADE",
            columns="ANO",
            values=outcome,
            aggfunc="first"
        )
        .reset_index()
    )

    wide.columns = [
        f"IDEB_{int(c)}" if isinstance(c, (int, float)) else c
        for c in wide.columns
    ]

    cols_info = [
        "CO_ENTIDADE",
        "tratamento_2017",
        "pct_bolsa_familia_2017",
        "ln_QT_MAT_FUND_2017",
        "urbana",
        "dep_estadual",
        "dep_municipal"
    ]

    cols_info = [c for c in cols_info if c in painel.columns]

    info = (
        painel[cols_info]
        .drop_duplicates("CO_ENTIDADE")
        .copy()
    )

    wide = wide.merge(info, on="CO_ENTIDADE", how="left")

    # Deltas usados para diagnóstico de balanceamento
    wide["delta_2009_2015"] = wide["IDEB_2015"] - wide["IDEB_2009"]
    wide["delta_2011_2015"] = wide["IDEB_2015"] - wide["IDEB_2011"]
    wide["delta_2013_2015"] = wide["IDEB_2015"] - wide["IDEB_2013"]

    return wide


def fazer_matching_caliper(wide, etapa):
    """
    Matching 1 vizinho com reposição, usando IDEB 2009, 2011, 2013 e 2015.
    Caliper aplicado sobre a distância euclidiana padronizada.
    """

    match_vars = [f"IDEB_{ano}" for ano in pre_years]

    base = wide.dropna(subset=match_vars + ["IDEB_2019", "tratamento_2017"]).copy()

    tratados = base[base["tratamento_2017"] == 1].copy()
    controles = base[base["tratamento_2017"] == 0].copy()

    print("\n==============================")
    print(f"MATCHING POR IDEB COM CALIPER {caliper} - {etapa}")
    print("==============================")
    print("Base wide:", base.shape)
    print("Tratadas elegíveis antes do caliper:", tratados.shape[0])
    print("Controles elegíveis:", controles.shape[0])
    print("Variáveis de matching:", match_vars)

    scaler = StandardScaler()

    X_controles = scaler.fit_transform(controles[match_vars])
    X_tratados = scaler.transform(tratados[match_vars])

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_controles)

    distances, indices = nn.kneighbors(X_tratados)

    pares = []

    controles_index = controles.index.to_numpy()
    tratados_index = tratados.index.to_numpy()

    for i in range(len(tratados_index)):
        dist = float(distances[i][0])
        idx_controle = controles_index[indices[i][0]]

        if dist <= caliper:
            pares.append({
                "CO_ENTIDADE_TRATADA": tratados.loc[tratados_index[i], "CO_ENTIDADE"],
                "CO_ENTIDADE_CONTROLE": controles.loc[idx_controle, "CO_ENTIDADE"],
                "distancia": dist
            })

    pares = pd.DataFrame(pares)

    tratadas_mantidas = pares["CO_ENTIDADE_TRATADA"].unique()
    controles_matched = pares["CO_ENTIDADE_CONTROLE"].unique()

    escolas_matched = pd.DataFrame({
        "CO_ENTIDADE": np.concatenate([tratadas_mantidas, controles_matched])
    })

    escolas_matched = escolas_matched.drop_duplicates()

    wide_matched = base.merge(escolas_matched, on="CO_ENTIDADE", how="inner")

    print("\nPares aceitos pelo caliper:", pares.shape[0])
    print("Tratadas mantidas após caliper:", len(tratadas_mantidas))
    print("Tratadas descartadas pelo caliper:", tratados.shape[0] - len(tratadas_mantidas))
    print("Controles únicos matched:", len(controles_matched))

    print("\nDistâncias dos pares aceitos:")
    print("Média:", pares["distancia"].mean())
    print("Mediana:", pares["distancia"].median())
    print("P90:", pares["distancia"].quantile(0.90))
    print("P95:", pares["distancia"].quantile(0.95))
    print("Máxima:", pares["distancia"].max())

    balance_vars = [
        "IDEB_2009",
        "IDEB_2011",
        "IDEB_2013",
        "IDEB_2015",
        "delta_2009_2015",
        "delta_2011_2015",
        "delta_2013_2015"
    ]

    balance = []
    balance += resumo_balanceamento(base, etapa, "antes_matching", balance_vars)
    balance += resumo_balanceamento(wide_matched, etapa, "depois_matching_caliper025", balance_vars)

    print("\nBalanceamento antes/depois:")
    print(pd.DataFrame(balance).to_string(index=False))

    return wide_matched, pares, balance


def montar_painel_matched(painel, wide_matched):
    escolas = wide_matched[["CO_ENTIDADE"]].drop_duplicates()
    painel_matched = painel.merge(escolas, on="CO_ENTIDADE", how="inner")
    return painel_matched


# ======================================================
# 4. Event study, pré-tendência e delta
# ======================================================

def rodar_eventstudy_e_pretrend(df, outcome, etapa):
    d = df.copy()
    d = d[d["ANO"].isin(anos_event_study)].copy()
    d = d.dropna(subset=[outcome, "tratamento_2017", "CO_ENTIDADE", "ANO"]).copy()

    print("\n==============================")
    print(f"EVENT STUDY 2009-2019 - {etapa}")
    print("==============================")
    print("Base usada:", d.shape)
    print("Escolas únicas:", d["CO_ENTIDADE"].nunique())
    print("Anos:")
    print(d["ANO"].value_counts().sort_index())

    xcols = []

    for ano in anos_event:
        col = f"tratado_x_{ano}"
        d[col] = ((d["tratamento_2017"] == 1) & (d["ANO"] == ano)).astype(int)
        xcols.append(col)

    cols_demean = [outcome] + xcols

    d_res = two_way_demean(
        d,
        cols=cols_demean,
        entity_col="CO_ENTIDADE",
        time_col="ANO"
    )

    y = d_res[outcome]
    X = d_res[xcols]

    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["CO_ENTIDADE"]}
    )

    print(model.summary())

    resultados = resumo_modelo(
        model=model,
        nome_modelo="event_study_twfe_matched_caliper025_base_2015",
        etapa=etapa,
        outcome=outcome,
        n_obs=d.shape[0],
        n_escolas=d["CO_ENTIDADE"].nunique()
    )

    # Teste conjunto de pré-tendência
    param_names = list(model.params.index)
    R = np.zeros((len(pre_test), len(param_names)))

    for i, var in enumerate(pre_test):
        j = param_names.index(var)
        R[i, j] = 1

    wald = model.wald_test(R, scalar=True)

    print("\n==============================")
    print(f"TESTE CONJUNTO DE PRÉ-TENDÊNCIA - {etapa}")
    print("==============================")
    print("H0: tratado_x_2009 = tratado_x_2011 = tratado_x_2013 = 0")
    print("Estatística:", float(wald.statistic))
    print("p-valor:", float(wald.pvalue))
    print("df restrições:", len(pre_test))

    resultados.append({
        "modelo": "teste_conjunto_pre_tendencia",
        "etapa": etapa,
        "outcome": outcome,
        "variavel": "tratado_x_2009 = tratado_x_2011 = tratado_x_2013 = 0",
        "coef": np.nan,
        "erro_padrao": np.nan,
        "t": np.nan,
        "pvalor": np.nan,
        "n_obs": d.shape[0],
        "n_escolas": d["CO_ENTIDADE"].nunique(),
        "r2": np.nan,
        "wald_stat_pretrend": float(wald.statistic),
        "wald_pvalor_pretrend": float(wald.pvalue),
        "wald_df": len(pre_test)
    })

    return resultados


def preparar_delta(df, outcome):
    d = df.copy()
    d = d[d["ANO"].isin([2015, 2019])].copy()
    d = d.dropna(subset=[outcome, "tratamento_2017"]).copy()

    wide_y = (
        d.pivot_table(
            index="CO_ENTIDADE",
            columns="ANO",
            values=outcome,
            aggfunc="first"
        )
        .reset_index()
    )

    wide_y = wide_y.rename(columns={
        2015: f"{outcome}_2015",
        2019: f"{outcome}_2019"
    })

    controles = [
        "CO_ENTIDADE",
        "tratamento_2017",
        "pct_bolsa_familia_2017",
        "ln_QT_MAT_FUND_2017",
        "urbana",
        "dep_estadual",
        "dep_municipal"
    ]

    controles = [c for c in controles if c in df.columns]

    base_controles = (
        df[controles]
        .drop_duplicates("CO_ENTIDADE")
        .copy()
    )

    out = wide_y.merge(base_controles, on="CO_ENTIDADE", how="left")

    out[f"delta_{outcome}_2015_2019"] = (
        out[f"{outcome}_2019"] - out[f"{outcome}_2015"]
    )

    out = out.dropna(subset=[
        f"{outcome}_2015",
        f"{outcome}_2019",
        f"delta_{outcome}_2015_2019",
        "tratamento_2017"
    ]).copy()

    return out


def rodar_delta_modelos(df_delta, outcome, etapa):
    resultados = []

    y_delta = f"delta_{outcome}_2015_2019"
    y_2015 = f"{outcome}_2015"

    formulas = {
        "did_delta_matched_caliper025_simples": (
            f"{y_delta} ~ tratamento_2017"
        ),
        "did_delta_matched_caliper025_controles": (
            f"{y_delta} ~ tratamento_2017 "
            f"+ pct_bolsa_familia_2017 "
            f"+ ln_QT_MAT_FUND_2017 "
            f"+ urbana + dep_estadual + dep_municipal"
        ),
        "did_delta_matched_caliper025_controles_baseline": (
            f"{y_delta} ~ tratamento_2017 "
            f"+ {y_2015} "
            f"+ pct_bolsa_familia_2017 "
            f"+ ln_QT_MAT_FUND_2017 "
            f"+ urbana + dep_estadual + dep_municipal"
        )
    }

    print("\n==============================")
    print(f"BASE DELTA 2015-2019 - {etapa}")
    print("==============================")
    print(df_delta.shape)
    print("Tratamento:")
    print(df_delta["tratamento_2017"].value_counts())

    for nome, formula in formulas.items():
        cols_formula = [y_delta, "tratamento_2017"]

        if "pct_bolsa_familia_2017" in formula:
            cols_formula += [
                "pct_bolsa_familia_2017",
                "ln_QT_MAT_FUND_2017",
                "urbana",
                "dep_estadual",
                "dep_municipal"
            ]

        if y_2015 in formula:
            cols_formula.append(y_2015)

        cols_formula = list(dict.fromkeys(cols_formula))

        d_model = df_delta.dropna(subset=cols_formula).copy()

        model = smf.ols(formula, data=d_model).fit(cov_type="HC1")

        print("\n==============================")
        print(f"{nome} - {etapa}")
        print("==============================")
        print(model.summary())

        resultados += resumo_modelo(
            model=model,
            nome_modelo=nome,
            etapa=etapa,
            outcome=outcome,
            n_obs=d_model.shape[0],
            n_escolas=d_model["CO_ENTIDADE"].nunique()
        )

    return resultados


def calcular_medias(df, outcome, etapa):
    d = df.dropna(subset=[outcome, "tratamento_2017"]).copy()

    medias = (
        d.groupby(["ANO", "tratamento_2017"])[outcome]
        .agg(["mean", "count", "std"])
        .reset_index()
    )

    medias["etapa"] = etapa
    medias["outcome"] = outcome

    return medias


# ======================================================
# 5. Rodar tudo
# ======================================================

print("\n==============================")
print("CARREGANDO PAINEL")
print("==============================")

painel = pd.read_csv(path_painel, dtype={"CO_ENTIDADE": str}, low_memory=False)

painel["CO_ENTIDADE"] = painel["CO_ENTIDADE"].astype(str).str.zfill(8)
painel["ANO"] = pd.to_numeric(painel["ANO"], errors="coerce").astype(int)
painel["tratamento_2017"] = pd.to_numeric(painel["tratamento_2017"], errors="coerce")

print(painel.shape)
print(painel["ANO"].value_counts().sort_index())

configs = [
    ("AI", "IDEB_AI"),
    ("AF", "IDEB_AF")
]

todos_resultados = []
todos_balanceamentos = []
todas_medias = []

for etapa, outcome in configs:
    print("\n\n################################")
    print(f"RODANDO MODELO PRINCIPAL - {etapa}")
    print("################################")

    wide = preparar_base_wide_para_matching(painel, etapa, outcome)

    wide_matched, pares, balance = fazer_matching_caliper(wide, etapa)

    todos_balanceamentos += balance

    # Salvar bases matched específicas
    path_escolas = out_dir + fr"\matched_escolas_IDEB_{etapa}_caliper025.csv"
    path_pares = out_dir + fr"\matched_escolas_IDEB_{etapa}_caliper025_pares.csv"
    path_painel_matched = out_dir + fr"\matched_painel_IDEB_{etapa}_caliper025.csv"

    wide_matched.to_csv(path_escolas, index=False, encoding="utf-8-sig")
    pares.to_csv(path_pares, index=False, encoding="utf-8-sig")

    painel_matched = montar_painel_matched(painel, wide_matched)
    painel_matched.to_csv(path_painel_matched, index=False, encoding="utf-8-sig")

    print("\nArquivos matched salvos:")
    print(path_escolas)
    print(path_pares)
    print(path_painel_matched)

    print("\nPainel matched:")
    print(painel_matched.shape)
    print("Escolas únicas:", painel_matched["CO_ENTIDADE"].nunique())
    print("Tratamento médio:", painel_matched["tratamento_2017"].mean())
    print("Anos:")
    print(painel_matched["ANO"].value_counts().sort_index())

    medias = calcular_medias(painel_matched, outcome, etapa)
    todas_medias.append(medias)

    print("\nMédias por ano e tratamento:")
    print(medias.to_string(index=False))

    resultados_event = rodar_eventstudy_e_pretrend(
        df=painel_matched,
        outcome=outcome,
        etapa=etapa
    )

    todos_resultados += resultados_event

    df_delta = preparar_delta(painel_matched, outcome)

    resultados_delta = rodar_delta_modelos(
        df_delta=df_delta,
        outcome=outcome,
        etapa=etapa
    )

    todos_resultados += resultados_delta


# ======================================================
# 6. Salvar resultados finais
# ======================================================

df_balance = pd.DataFrame(todos_balanceamentos)
df_resultados = pd.DataFrame(todos_resultados)
df_medias = pd.concat(todas_medias, ignore_index=True)

df_balance.to_csv(output_balanceamento, index=False, encoding="utf-8-sig")
df_resultados.to_csv(output_resultados, index=False, encoding="utf-8-sig")
df_medias.to_csv(output_medias, index=False, encoding="utf-8-sig")

print("\n==============================")
print("ARQUIVOS FINAIS SALVOS")
print("==============================")
print(output_balanceamento)
print(output_resultados)
print(output_medias)

print("\nResumo dos coeficientes principais:")
principais = df_resultados[
    df_resultados["variavel"].isin([
        "tratamento_2017",
        "tratado_x_2009",
        "tratado_x_2011",
        "tratado_x_2013",
        "tratado_x_2017",
        "tratado_x_2019",
        "tratado_x_2009 = tratado_x_2011 = tratado_x_2013 = 0"
    ])
].copy()

print(principais.to_string(index=False))

print("\nResumo do teste conjunto de pré-tendência:")
pretrend = df_resultados[
    df_resultados["modelo"] == "teste_conjunto_pre_tendencia"
].copy()

print(
    pretrend[
        [
            "etapa",
            "outcome",
            "wald_stat_pretrend",
            "wald_pvalor_pretrend",
            "wald_df"
        ]
    ].to_string(index=False)
)
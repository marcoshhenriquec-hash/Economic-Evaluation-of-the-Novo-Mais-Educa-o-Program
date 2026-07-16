# ======================================================
# run_DiD_matched_caliper025_IDEB.py
# DiD/event study na amostra matched por IDEB 2009-2015
# com caliper 0.25
# ======================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# -------------------------------
# 1. Caminhos
# -------------------------------

path_ai = r"C:\Users\suporte1\Desktop\lai\matched_painel_IDEB_AI_caliper025.csv"
path_af = r"C:\Users\suporte1\Desktop\lai\matched_painel_IDEB_AF_caliper025.csv"

output_resultados = r"C:\Users\suporte1\Desktop\lai\resultados_DiD_eventstudy_matched_IDEB_caliper025.csv"
output_medias = r"C:\Users\suporte1\Desktop\lai\medias_matched_IDEB_tratamento_controle_caliper025.csv"

anos = [2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019]
anos_event = [2005, 2007, 2009, 2011, 2013, 2017, 2019]  # 2015 é base


# -------------------------------
# 2. Funções auxiliares
# -------------------------------

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
            "r2": getattr(model, "rsquared", np.nan)
        })

    return linhas


def two_way_demean(data, cols, entity_col="CO_ENTIDADE", time_col="ANO", max_iter=50, tol=1e-10):
    """
    Residualiza variáveis em relação a efeitos fixos de escola e ano.
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


def rodar_eventstudy_twfe(df_etapa, outcome, etapa):
    """
    Event study com efeitos fixos de escola e ano.
    Ano-base = 2015.
    """
    d = df_etapa.copy()
    d = d[d["ANO"].isin(anos)].copy()
    d = d.dropna(subset=[outcome, "tratamento_2017", "CO_ENTIDADE", "ANO"]).copy()

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

    print("\n==============================")
    print(f"EVENT STUDY TWFE MATCHED CALIPER 0.25 - {etapa}")
    print("==============================")
    print(model.summary())

    return resumo_modelo(
        model=model,
        nome_modelo="event_study_twfe_matched_caliper025_base_2015",
        etapa=etapa,
        outcome=outcome,
        n_obs=d.shape[0],
        n_escolas=d["CO_ENTIDADE"].nunique()
    )


def preparar_delta(df, outcome):
    """
    Cria base 2015-2019 com uma linha por escola.
    """
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
    """
    Roda DiD via delta:
    1. simples
    2. com controles
    3. com controles + IDEB 2015
    """
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


# -------------------------------
# 3. Rodar modelos
# -------------------------------

configs = [
    ("AI", "IDEB_AI", path_ai),
    ("AF", "IDEB_AF", path_af)
]

todos_resultados = []
todas_medias = []

for etapa, outcome, path in configs:
    print("\n\n################################")
    print(f"RODANDO MATCHED CALIPER 0.25: {etapa}")
    print("################################")

    df = pd.read_csv(
        path,
        dtype={"CO_ENTIDADE": str},
        low_memory=False
    )

    df["CO_ENTIDADE"] = df["CO_ENTIDADE"].astype(str).str.zfill(8)
    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").astype(int)
    df["tratamento_2017"] = pd.to_numeric(df["tratamento_2017"], errors="coerce")

    print("\nBase matched caliper:")
    print(df.shape)
    print("Escolas únicas:", df["CO_ENTIDADE"].nunique())
    print("Tratamento médio:", df["tratamento_2017"].mean())
    print("Anos:")
    print(df["ANO"].value_counts().sort_index())

    medias = calcular_medias(df, outcome, etapa)
    todas_medias.append(medias)

    print("\nMédias matched caliper por ano e tratamento:")
    print(medias)

    resultados_event = rodar_eventstudy_twfe(
        df_etapa=df,
        outcome=outcome,
        etapa=etapa
    )

    todos_resultados += resultados_event

    df_delta = preparar_delta(df, outcome)

    print("\n==============================")
    print(f"BASE DELTA MATCHED CALIPER 2015-2019 - {etapa}")
    print("==============================")
    print(df_delta.shape)
    print(df_delta[[
        "CO_ENTIDADE",
        "tratamento_2017",
        f"{outcome}_2015",
        f"{outcome}_2019",
        f"delta_{outcome}_2015_2019"
    ]].head())

    print("\nTratamento na base delta matched caliper:")
    print(df_delta["tratamento_2017"].value_counts())

    resultados_delta = rodar_delta_modelos(
        df_delta=df_delta,
        outcome=outcome,
        etapa=etapa
    )

    todos_resultados += resultados_delta


# -------------------------------
# 4. Salvar resultados
# -------------------------------

df_resultados = pd.DataFrame(todos_resultados)
df_medias = pd.concat(todas_medias, ignore_index=True)

df_resultados.to_csv(output_resultados, index=False, encoding="utf-8-sig")
df_medias.to_csv(output_medias, index=False, encoding="utf-8-sig")

print("\n==============================")
print("ARQUIVOS SALVOS")
print("==============================")
print(output_resultados)
print(output_medias)

print("\nResumo dos coeficientes principais matched caliper 0.25:")
principais = df_resultados[
    df_resultados["variavel"].isin([
        "tratamento_2017",
        "tratado_x_2005",
        "tratado_x_2007",
        "tratado_x_2009",
        "tratado_x_2011",
        "tratado_x_2013",
        "tratado_x_2017",
        "tratado_x_2019"
    ])
].copy()

print(principais.to_string(index=False))
# ======================================================
# teste_pre_tendencia_caliper025.py
# Teste conjunto de pré-tendência: 2009, 2011, 2013
# Amostra matched por IDEB 2009-2015 com caliper 0.25
# ======================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm

# -------------------------------
# 1. Caminhos
# -------------------------------

path_ai = r"C:\Users\suporte1\Desktop\lai\matched_painel_IDEB_AI_caliper025.csv"
path_af = r"C:\Users\suporte1\Desktop\lai\matched_painel_IDEB_AF_caliper025.csv"

output_resultados = r"C:\Users\suporte1\Desktop\lai\teste_pre_tendencia_caliper025.csv"

anos = [2009, 2011, 2013, 2015, 2017, 2019]

# 2015 é o ano-base, então não entra como interação
anos_event = [2009, 2011, 2013, 2017, 2019]

# Pré-tendência a testar
pre_test = ["tratado_x_2009", "tratado_x_2011", "tratado_x_2013"]


# -------------------------------
# 2. Funções
# -------------------------------

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


def rodar_event_study_e_teste(path, etapa, outcome):
    print("\n\n################################")
    print(f"TESTE DE PRÉ-TENDÊNCIA - {etapa}")
    print("################################")

    df = pd.read_csv(
        path,
        dtype={"CO_ENTIDADE": str},
        low_memory=False
    )

    df["CO_ENTIDADE"] = df["CO_ENTIDADE"].astype(str).str.zfill(8)
    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").astype(int)
    df["tratamento_2017"] = pd.to_numeric(df["tratamento_2017"], errors="coerce")

    d = df[df["ANO"].isin(anos)].copy()
    d = d.dropna(subset=[outcome, "tratamento_2017", "CO_ENTIDADE", "ANO"]).copy()

    print("\nBase usada:")
    print(d.shape)
    print("Escolas únicas:", d["CO_ENTIDADE"].nunique())
    print("Anos:")
    print(d["ANO"].value_counts().sort_index())

    # Criar interações tratado x ano
    xcols = []

    for ano in anos_event:
        col = f"tratado_x_{ano}"
        d[col] = ((d["tratamento_2017"] == 1) & (d["ANO"] == ano)).astype(int)
        xcols.append(col)

    # Residualizar por escola e ano
    cols_demean = [outcome] + xcols

    d_res = two_way_demean(
        d,
        cols=cols_demean,
        entity_col="CO_ENTIDADE",
        time_col="ANO"
    )

    y = d_res[outcome]
    X = d_res[xcols]

    # OLS sem constante após two-way demeaning
    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["CO_ENTIDADE"]}
    )

    print("\n==============================")
    print(f"EVENT STUDY 2009-2019 - {etapa}")
    print("==============================")
    print(model.summary())

    # Teste conjunto: tratado_x_2009 = tratado_x_2011 = tratado_x_2013 = 0
    # Monta matriz R para Wald test
    R = np.zeros((len(pre_test), len(model.params)))
    param_names = list(model.params.index)

    for i, var in enumerate(pre_test):
        if var not in param_names:
            raise ValueError(f"Variável {var} não encontrada no modelo.")
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

    if float(wald.pvalue) >= 0.05:
        interpretacao = "Não rejeita H0: não há evidência estatística forte de pré-tendência diferencial em 2009-2015."
    else:
        interpretacao = "Rejeita H0: há evidência de pré-tendência diferencial em 2009-2015."

    print("Interpretação:", interpretacao)

    # Salvar coeficientes e resultado do teste
    linhas = []

    for var in model.params.index:
        linhas.append({
            "etapa": etapa,
            "outcome": outcome,
            "tipo": "coeficiente_event_study",
            "variavel": var,
            "coef": model.params[var],
            "erro_padrao": model.bse[var],
            "t": model.tvalues[var],
            "pvalor": model.pvalues[var],
            "n_obs": d.shape[0],
            "n_escolas": d["CO_ENTIDADE"].nunique(),
            "wald_stat_pretrend": np.nan,
            "wald_pvalor_pretrend": np.nan,
            "wald_df": np.nan,
            "interpretacao": ""
        })

    linhas.append({
        "etapa": etapa,
        "outcome": outcome,
        "tipo": "teste_conjunto_pre_tendencia",
        "variavel": "tratado_x_2009 = tratado_x_2011 = tratado_x_2013 = 0",
        "coef": np.nan,
        "erro_padrao": np.nan,
        "t": np.nan,
        "pvalor": np.nan,
        "n_obs": d.shape[0],
        "n_escolas": d["CO_ENTIDADE"].nunique(),
        "wald_stat_pretrend": float(wald.statistic),
        "wald_pvalor_pretrend": float(wald.pvalue),
        "wald_df": len(pre_test),
        "interpretacao": interpretacao
    })

    return linhas


# -------------------------------
# 3. Rodar AI e AF
# -------------------------------

todos = []

todos += rodar_event_study_e_teste(
    path=path_ai,
    etapa="AI",
    outcome="IDEB_AI"
)

todos += rodar_event_study_e_teste(
    path=path_af,
    etapa="AF",
    outcome="IDEB_AF"
)

df_resultados = pd.DataFrame(todos)
df_resultados.to_csv(output_resultados, index=False, encoding="utf-8-sig")

print("\n==============================")
print("RESULTADOS SALVOS")
print("==============================")
print(output_resultados)

print("\nResumo teste conjunto:")
print(
    df_resultados[df_resultados["tipo"] == "teste_conjunto_pre_tendencia"][
        [
            "etapa",
            "outcome",
            "variavel",
            "wald_stat_pretrend",
            "wald_pvalor_pretrend",
            "wald_df",
            "interpretacao"
        ]
    ].to_string(index=False)
)
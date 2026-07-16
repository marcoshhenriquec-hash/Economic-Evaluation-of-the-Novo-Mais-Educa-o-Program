# ======================================================
# reg_ideb_2015_2019.py
# Regressões IDEB com baseline pré-PNME: 2015-2019
# Modelo principal: IDEB 2019
# Robustez: delta IDEB 2015-2019
# ======================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# -------------------------------
# 1. Caminhos
# -------------------------------

path_base = r"C:\Users\suporte1\Desktop\lai\base_modelo_ideb_2015_2019.csv"

output_resumo = r"C:\Users\suporte1\Desktop\lai\resultados_reg_ideb_2015_2019.csv"

# -------------------------------
# 2. Ler base
# -------------------------------

df = pd.read_csv(
    path_base,
    dtype={"CO_ENTIDADE": str},
    low_memory=False
)

df.columns = df.columns.str.strip()

# -------------------------------
# 3. Ajustar tipos
# -------------------------------

cols_num = [
    "tratamento_2017",
    "pct_bolsa_familia_2017",
    "corte_50_2017",
    "QT_MAT_FUND_2017",
    "urbana",
    "dep_estadual",
    "dep_municipal",

    "IDEB_AI_2015",
    "IDEB_AI_2017",
    "IDEB_AI_2019",
    "IDEB_AF_2015",
    "IDEB_AF_2017",
    "IDEB_AF_2019",

    "delta_IDEB_AI_2015_2019",
    "delta_IDEB_AF_2015_2019",
    "delta_IDEB_AI_2017_2019",
    "delta_IDEB_AF_2017_2019"
]

for col in cols_num:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------------
# 4. Criar log do tamanho da escola
# -------------------------------

df["ln_QT_MAT_FUND_2017"] = np.where(
    df["QT_MAT_FUND_2017"] > 0,
    np.log(df["QT_MAT_FUND_2017"]),
    np.nan
)

# -------------------------------
# 5. Amostras AI e AF
# -------------------------------

controles_base = [
    "tratamento_2017",
    "pct_bolsa_familia_2017",
    "ln_QT_MAT_FUND_2017",
    "urbana",
    "dep_estadual",
    "dep_municipal"
]

df_ai = df.dropna(
    subset=[
        "IDEB_AI_2015",
        "IDEB_AI_2019",
        "delta_IDEB_AI_2015_2019"
    ] + controles_base
).copy()

df_af = df.dropna(
    subset=[
        "IDEB_AF_2015",
        "IDEB_AF_2019",
        "delta_IDEB_AF_2015_2019"
    ] + controles_base
).copy()

print("\n==============================")
print("AMOSTRAS")
print("==============================")
print("AI:", df_ai.shape)
print("AF:", df_af.shape)

print("\nTratamento AI:")
print(df_ai["tratamento_2017"].value_counts())

print("\nTratamento AF:")
print(df_af["tratamento_2017"].value_counts())

# -------------------------------
# 6. Função auxiliar
# -------------------------------

def rodar_modelo(nome, formula, data):
    print("\n==============================")
    print(nome)
    print("==============================")

    reg = smf.ols(
        formula=formula,
        data=data
    ).fit(cov_type="HC1")

    print(reg.summary())

    return reg

# ======================================================
# 7. MODELOS PRINCIPAIS - IDEB 2019
# ======================================================

# -------------------------------
# AI - anos iniciais
# -------------------------------

reg_ai_1 = rodar_modelo(
    "REG 1 - AI simples",
    "IDEB_AI_2019 ~ tratamento_2017",
    df_ai
)

reg_ai_2 = rodar_modelo(
    "REG 2 - AI + IDEB 2015",
    """
    IDEB_AI_2019 ~ tratamento_2017
                 + IDEB_AI_2015
    """,
    df_ai
)

reg_ai_3 = rodar_modelo(
    "REG 3 - AI + IDEB 2015 + BF",
    """
    IDEB_AI_2019 ~ tratamento_2017
                 + IDEB_AI_2015
                 + pct_bolsa_familia_2017
    """,
    df_ai
)

reg_ai_4 = rodar_modelo(
    "REG 4 - AI + IDEB 2015 + BF + tamanho",
    """
    IDEB_AI_2019 ~ tratamento_2017
                 + IDEB_AI_2015
                 + pct_bolsa_familia_2017
                 + ln_QT_MAT_FUND_2017
    """,
    df_ai
)

reg_ai_5 = rodar_modelo(
    "REG 5 - AI completo",
    """
    IDEB_AI_2019 ~ tratamento_2017
                 + IDEB_AI_2015
                 + pct_bolsa_familia_2017
                 + ln_QT_MAT_FUND_2017
                 + urbana
                 + dep_estadual
                 + dep_municipal
    """,
    df_ai
)

# -------------------------------
# AF - anos finais
# -------------------------------

reg_af_1 = rodar_modelo(
    "REG 6 - AF simples",
    "IDEB_AF_2019 ~ tratamento_2017",
    df_af
)

reg_af_2 = rodar_modelo(
    "REG 7 - AF + IDEB 2015",
    """
    IDEB_AF_2019 ~ tratamento_2017
                 + IDEB_AF_2015
    """,
    df_af
)

reg_af_3 = rodar_modelo(
    "REG 8 - AF + IDEB 2015 + BF",
    """
    IDEB_AF_2019 ~ tratamento_2017
                 + IDEB_AF_2015
                 + pct_bolsa_familia_2017
    """,
    df_af
)

reg_af_4 = rodar_modelo(
    "REG 9 - AF + IDEB 2015 + BF + tamanho",
    """
    IDEB_AF_2019 ~ tratamento_2017
                 + IDEB_AF_2015
                 + pct_bolsa_familia_2017
                 + ln_QT_MAT_FUND_2017
    """,
    df_af
)

reg_af_5 = rodar_modelo(
    "REG 10 - AF completo",
    """
    IDEB_AF_2019 ~ tratamento_2017
                 + IDEB_AF_2015
                 + pct_bolsa_familia_2017
                 + ln_QT_MAT_FUND_2017
                 + urbana
                 + dep_estadual
                 + dep_municipal
    """,
    df_af
)

# ======================================================
# 8. ROBUSTEZ - DELTA 2015-2019
# ======================================================

reg_ai_delta = rodar_modelo(
    "REG 11 - Robustez AI delta 2015-2019",
    """
    delta_IDEB_AI_2015_2019 ~ tratamento_2017
                            + IDEB_AI_2015
                            + pct_bolsa_familia_2017
                            + ln_QT_MAT_FUND_2017
                            + urbana
                            + dep_estadual
                            + dep_municipal
    """,
    df_ai
)

reg_af_delta = rodar_modelo(
    "REG 12 - Robustez AF delta 2015-2019",
    """
    delta_IDEB_AF_2015_2019 ~ tratamento_2017
                            + IDEB_AF_2015
                            + pct_bolsa_familia_2017
                            + ln_QT_MAT_FUND_2017
                            + urbana
                            + dep_estadual
                            + dep_municipal
    """,
    df_af
)

# ======================================================
# 9. MODELO CONJUNTO AI + AF
# ======================================================

# Transformar AI e AF para formato long

df_ai_long = df_ai[[
    "CO_ENTIDADE",
    "tratamento_2017",
    "pct_bolsa_familia_2017",
    "ln_QT_MAT_FUND_2017",
    "urbana",
    "dep_estadual",
    "dep_municipal",
    "IDEB_AI_2015",
    "IDEB_AI_2019"
]].copy()

df_ai_long = df_ai_long.rename(columns={
    "IDEB_AI_2015": "IDEB_2015",
    "IDEB_AI_2019": "IDEB_2019"
})

df_ai_long["etapa"] = "AI"

df_af_long = df_af[[
    "CO_ENTIDADE",
    "tratamento_2017",
    "pct_bolsa_familia_2017",
    "ln_QT_MAT_FUND_2017",
    "urbana",
    "dep_estadual",
    "dep_municipal",
    "IDEB_AF_2015",
    "IDEB_AF_2019"
]].copy()

df_af_long = df_af_long.rename(columns={
    "IDEB_AF_2015": "IDEB_2015",
    "IDEB_AF_2019": "IDEB_2019"
})

df_af_long["etapa"] = "AF"

df_long = pd.concat([df_ai_long, df_af_long], ignore_index=True)

print("\n==============================")
print("AMOSTRA CONJUNTA AI + AF")
print("==============================")
print(df_long.shape)
print(df_long["etapa"].value_counts())

reg_conjunto_1 = rodar_modelo(
    "REG 13 - Conjunto AI + AF com dummy de etapa",
    """
    IDEB_2019 ~ tratamento_2017
              + IDEB_2015
              + pct_bolsa_familia_2017
              + ln_QT_MAT_FUND_2017
              + urbana
              + dep_estadual
              + dep_municipal
              + C(etapa)
    """,
    df_long
)

reg_conjunto_2 = rodar_modelo(
    "REG 14 - Conjunto AI + AF com interação tratamento x etapa",
    """
    IDEB_2019 ~ tratamento_2017 * C(etapa)
              + IDEB_2015
              + pct_bolsa_familia_2017
              + ln_QT_MAT_FUND_2017
              + urbana
              + dep_estadual
              + dep_municipal
              + C(etapa)
    """,
    df_long
)

# ======================================================
# 10. Resumo compacto
# ======================================================

modelos = [
    ("AI simples", reg_ai_1, "principal"),
    ("AI + IDEB 2015", reg_ai_2, "principal"),
    ("AI + IDEB 2015 + BF", reg_ai_3, "principal"),
    ("AI + IDEB 2015 + BF + tamanho", reg_ai_4, "principal"),
    ("AI completo", reg_ai_5, "principal"),

    ("AF simples", reg_af_1, "principal"),
    ("AF + IDEB 2015", reg_af_2, "principal"),
    ("AF + IDEB 2015 + BF", reg_af_3, "principal"),
    ("AF + IDEB 2015 + BF + tamanho", reg_af_4, "principal"),
    ("AF completo", reg_af_5, "principal"),

    ("AI delta completo", reg_ai_delta, "robustez_delta"),
    ("AF delta completo", reg_af_delta, "robustez_delta"),

    ("Conjunto AI + AF", reg_conjunto_1, "conjunto"),
    ("Conjunto AI + AF com interação", reg_conjunto_2, "conjunto_interacao"),
]

linhas = []

for nome, reg, tipo in modelos:
    linha = {
        "modelo": nome,
        "tipo": tipo,
        "coef_tratamento": reg.params.get("tratamento_2017"),
        "erro_padrao": reg.bse.get("tratamento_2017"),
        "p_valor": reg.pvalues.get("tratamento_2017"),
        "n_obs": int(reg.nobs),
        "r2": reg.rsquared,
        "controles": ", ".join(
            [v for v in reg.params.index if v not in ["Intercept", "tratamento_2017"]]
        )
    }

    # Só aparece no modelo com interação
    linha["coef_interacao_AF"] = reg.params.get("tratamento_2017:C(etapa)[T.AF]")
    linha["p_interacao_AF"] = reg.pvalues.get("tratamento_2017:C(etapa)[T.AF]")

    linhas.append(linha)

resumo = pd.DataFrame(linhas)

print("\n==============================")
print("RESUMO DOS COEFICIENTES DE TRATAMENTO")
print("==============================")
print(resumo)

# -------------------------------
# 11. Salvar resumo
# -------------------------------

resumo.to_csv(output_resumo, index=False, encoding="utf-8-sig")

print("\nResumo salvo em:")
print(output_resumo)
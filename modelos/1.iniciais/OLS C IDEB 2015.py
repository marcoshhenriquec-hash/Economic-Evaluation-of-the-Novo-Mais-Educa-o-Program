#OLS C IDEB 2015

# ======================================================
# reg_ideb_2015_2019.py
# Regressões IDEB com baseline pré-PNME: 2015-2019
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
# 5. Amostras analíticas
# -------------------------------

df_ai = df.dropna(
    subset=[
        "IDEB_AI_2015",
        "IDEB_AI_2019",
        "delta_IDEB_AI_2015_2019",
        "tratamento_2017",
        "pct_bolsa_familia_2017",
        "ln_QT_MAT_FUND_2017",
        "urbana",
        "dep_estadual",
        "dep_municipal"
    ]
).copy()

df_af = df.dropna(
    subset=[
        "IDEB_AF_2015",
        "IDEB_AF_2019",
        "delta_IDEB_AF_2015_2019",
        "tratamento_2017",
        "pct_bolsa_familia_2017",
        "ln_QT_MAT_FUND_2017",
        "urbana",
        "dep_estadual",
        "dep_municipal"
    ]
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

# -------------------------------
# 7. Regressões AI - IDEB 2019
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
# 8. Regressões AF - IDEB 2019
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

# -------------------------------
# 9. Regressões com delta 2015-2019
# -------------------------------

reg_ai_delta = rodar_modelo(
    "REG 11 - AI delta 2015-2019 completo",
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
    "REG 12 - AF delta 2015-2019 completo",
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

# -------------------------------
# 10. Resumo compacto
# -------------------------------

modelos = [
    ("AI simples", reg_ai_1),
    ("AI + IDEB 2015", reg_ai_2),
    ("AI + IDEB 2015 + BF", reg_ai_3),
    ("AI + IDEB 2015 + BF + tamanho", reg_ai_4),
    ("AI completo", reg_ai_5),

    ("AF simples", reg_af_1),
    ("AF + IDEB 2015", reg_af_2),
    ("AF + IDEB 2015 + BF", reg_af_3),
    ("AF + IDEB 2015 + BF + tamanho", reg_af_4),
    ("AF completo", reg_af_5),

    ("AI delta completo", reg_ai_delta),
    ("AF delta completo", reg_af_delta),
]

linhas = []

for nome, reg in modelos:
    linhas.append({
        "modelo": nome,
        "coef_tratamento": reg.params.get("tratamento_2017"),
        "erro_padrao": reg.bse.get("tratamento_2017"),
        "p_valor": reg.pvalues.get("tratamento_2017"),
        "n_obs": int(reg.nobs),
        "r2": reg.rsquared,
        "controles": ", ".join(
            [v for v in reg.params.index if v not in ["Intercept", "tratamento_2017"]]
        )
    })

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
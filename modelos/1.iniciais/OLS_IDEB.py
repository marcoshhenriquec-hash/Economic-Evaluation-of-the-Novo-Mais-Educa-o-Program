# ======================================================
# reg_delta_ideb.py
# Regressões delta IDEB 2017-2019
# ======================================================

import pandas as pd
import statsmodels.formula.api as smf

# -------------------------------
# 1. Caminho
# -------------------------------

path_delta = r"C:\Users\suporte1\Desktop\lai\base_delta_ideb_2017_2019.csv"

# -------------------------------
# 2. Ler base
# -------------------------------

df = pd.read_csv(
    path_delta,
    dtype={"CO_ENTIDADE": str},
    low_memory=False
)

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
    "IDEB_AI_2017",
    "IDEB_AI_2019",
    "delta_IDEB_AI",
    "IDEB_AF_2017",
    "IDEB_AF_2019",
    "delta_IDEB_AF"
]

for col in cols_num:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------------
# 4. Criar log do tamanho da escola
# -------------------------------

df["ln_QT_MAT_FUND_2017"] = (
    df["QT_MAT_FUND_2017"]
    .where(df["QT_MAT_FUND_2017"] > 0)
)

df["ln_QT_MAT_FUND_2017"] = df["ln_QT_MAT_FUND_2017"].apply(
    lambda x: pd.NA if pd.isna(x) else __import__("math").log(x)
)

df["ln_QT_MAT_FUND_2017"] = pd.to_numeric(
    df["ln_QT_MAT_FUND_2017"],
    errors="coerce"
)

# -------------------------------
# 5. Amostras analíticas
# -------------------------------

df_ai = df.dropna(
    subset=[
        "delta_IDEB_AI",
        "IDEB_AI_2017",
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
        "delta_IDEB_AF",
        "IDEB_AF_2017",
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
# 6. Regressões simples
# -------------------------------

print("\n==============================")
print("REG 1 - AI simples")
print("==============================")

reg_ai_1 = smf.ols(
    formula="IDEB_AI_2019 ~ tratamento_2017",
    data=df_ai
).fit(cov_type="HC1")

print(reg_ai_1.summary())


print("\n==============================")
print("REG 2 - AF simples")
print("==============================")

reg_af_1 = smf.ols(
    formula="IDEB_AF_2019 ~ tratamento_2017",
    data=df_af
).fit(cov_type="HC1")

print(reg_af_1.summary())

# -------------------------------
# 7. Regressões com controles
# -------------------------------

print("\n==============================")
print("REG 3 - AI com controles")
print("==============================")

reg_ai_2 = smf.ols(
    formula="""
    IDEB_AI_2019 ~ tratamento_2017
                 + IDEB_AI_2017
                 + pct_bolsa_familia_2017
                 + ln_QT_MAT_FUND_2017
    """,
    data=df_ai
).fit(cov_type="HC1")

print(reg_ai_2.summary())


print("\n==============================")
print("REG 4 - AF com controles")
print("==============================")

reg_af_2 = smf.ols(
    formula="""
    IDEB_AF_2019 ~ tratamento_2017
                 + IDEB_AF_2017
                 + pct_bolsa_familia_2017
                 + ln_QT_MAT_FUND_2017
                
                
    """,
    data=df_af
).fit(cov_type="HC1")

print(reg_af_2.summary())

# -------------------------------
# 8. Resumo compacto
# -------------------------------

print("\n==============================")
print("RESUMO DOS COEFICIENTES DE TRATAMENTO")
print("==============================")

resumo = pd.DataFrame({
    "modelo": [
        "AI simples",
        "AF simples",
        "AI controles",
        "AF controles"
    ],
    "coef_tratamento": [
        reg_ai_1.params.get("tratamento_2017"),
        reg_af_1.params.get("tratamento_2017"),
        reg_ai_2.params.get("tratamento_2017"),
        reg_af_2.params.get("tratamento_2017")
    ],
    "erro_padrao": [
        reg_ai_1.bse.get("tratamento_2017"),
        reg_af_1.bse.get("tratamento_2017"),
        reg_ai_2.bse.get("tratamento_2017"),
        reg_af_2.bse.get("tratamento_2017")
    ],
    "p_valor": [
        reg_ai_1.pvalues.get("tratamento_2017"),
        reg_af_1.pvalues.get("tratamento_2017"),
        reg_ai_2.pvalues.get("tratamento_2017"),
        reg_af_2.pvalues.get("tratamento_2017")
    ],
    "n_obs": [
        int(reg_ai_1.nobs),
        int(reg_af_1.nobs),
        int(reg_ai_2.nobs),
        int(reg_af_2.nobs)
    ],
    "r2": [
        reg_ai_1.rsquared,
        reg_af_1.rsquared,
        reg_ai_2.rsquared,
        reg_af_2.rsquared
    ]
})

print(resumo)

# -------------------------------
# 9. Salvar resumo
# -------------------------------

output_resumo = r"C:\Users\suporte1\Desktop\lai\resultados_reg_delta_ideb.csv"

resumo.to_csv(output_resumo, index=False, encoding="utf-8-sig")

print("\nResumo salvo em:")
print(output_resumo)
# ======================================================
# eda_ideb.py
# EDA da base RDD + PNME + IDEB
# ======================================================

import pandas as pd

# -------------------------------
# 1. Caminho
# -------------------------------

path_base = r"C:\Users\suporte1\Desktop\lai\base_rdd_pnme_ideb_2017_2019.csv"

# -------------------------------
# 2. Ler base
# -------------------------------

df = pd.read_csv(
    path_base,
    dtype={"CO_ENTIDADE": str},
    low_memory=False
)

# -------------------------------
# 3. Ajustar tipos
# -------------------------------

df["ANO"] = df["ANO"].astype(int)
df["tratamento_real"] = df["tratamento_real"].astype(int)
df["corte_50"] = df["corte_50"].astype(int)
df["pct_bolsa_familia"] = df["pct_bolsa_familia"].astype(float)

df["IDEB_AI"] = pd.to_numeric(df["IDEB_AI"], errors="coerce")
df["IDEB_AF"] = pd.to_numeric(df["IDEB_AF"], errors="coerce")

# -------------------------------
# 4. Checks gerais
# -------------------------------

print("\n==============================")
print("1. DIMENSÃO")
print("==============================")
print(df.shape)

print("\n==============================")
print("2. COLUNAS")
print("==============================")
print(df.columns.tolist())

print("\n==============================")
print("3. DUPLICATAS CO_ENTIDADE + ANO")
print("==============================")
print(df.duplicated(["CO_ENTIDADE", "ANO"]).sum())

print("\n==============================")
print("4. ANOS")
print("==============================")
print(df["ANO"].value_counts().sort_index())

print("\n==============================")
print("5. DISTRIBUIÇÃO tratamento_real")
print("==============================")
print(df["tratamento_real"].value_counts(dropna=False))

print("\n==============================")
print("6. TRATAMENTO POR ANO")
print("==============================")
print(pd.crosstab(df["ANO"], df["tratamento_real"]))

# -------------------------------
# 5. Disponibilidade do IDEB
# -------------------------------

print("\n==============================")
print("7. MISSING IDEB POR ANO")
print("==============================")
print(
    df.groupby("ANO")[["IDEB_AI", "IDEB_AF"]]
      .apply(lambda x: x.isna().mean())
)

print("\n==============================")
print("8. QUANTIDADE COM IDEB POR ANO")
print("==============================")
print(
    df.groupby("ANO")[["IDEB_AI", "IDEB_AF"]]
      .apply(lambda x: x.notna().sum())
)

# -------------------------------
# 6. Estatísticas das notas IDEB
# -------------------------------

print("\n==============================")
print("9. RESUMO IDEB_AI POR ANO")
print("==============================")
print(df.groupby("ANO")["IDEB_AI"].describe())

print("\n==============================")
print("10. RESUMO IDEB_AF POR ANO")
print("==============================")
print(df.groupby("ANO")["IDEB_AF"].describe())

print("\n==============================")
print("11. MÉDIA IDEB_AI POR ANO E TRATAMENTO")
print("==============================")
print(df.groupby(["ANO", "tratamento_real"])["IDEB_AI"].mean())

print("\n==============================")
print("12. MÉDIA IDEB_AF POR ANO E TRATAMENTO")
print("==============================")
print(df.groupby(["ANO", "tratamento_real"])["IDEB_AF"].mean())

print("\n==============================")
print("13. RESUMO IDEB_AI POR ANO E TRATAMENTO")
print("==============================")
print(df.groupby(["ANO", "tratamento_real"])["IDEB_AI"].describe())

print("\n==============================")
print("14. RESUMO IDEB_AF POR ANO E TRATAMENTO")
print("==============================")
print(df.groupby(["ANO", "tratamento_real"])["IDEB_AF"].describe())

print("\n==============================")
print("15. MÉDIA IDEB_AI POR ANO E CORTE_50")
print("==============================")
print(df.groupby(["ANO", "corte_50"])["IDEB_AI"].mean())

print("\n==============================")
print("16. MÉDIA IDEB_AF POR ANO E CORTE_50")
print("==============================")
print(df.groupby(["ANO", "corte_50"])["IDEB_AF"].mean())

# -------------------------------
# 7. Criar deltas 2017-2019
# -------------------------------

df_ideb = df[df["ANO"].isin([2017, 2019])].copy()

# IDEB_AI wide
df_ai_delta = (
    df_ideb
    .pivot_table(
        index="CO_ENTIDADE",
        columns="ANO",
        values="IDEB_AI",
        aggfunc="first"
    )
    .reset_index()
)

df_ai_delta = df_ai_delta.rename(columns={
    2017: "IDEB_AI_2017",
    2019: "IDEB_AI_2019"
})

df_ai_delta["delta_IDEB_AI"] = (
    df_ai_delta["IDEB_AI_2019"] - df_ai_delta["IDEB_AI_2017"]
)

# IDEB_AF wide
df_af_delta = (
    df_ideb
    .pivot_table(
        index="CO_ENTIDADE",
        columns="ANO",
        values="IDEB_AF",
        aggfunc="first"
    )
    .reset_index()
)

df_af_delta = df_af_delta.rename(columns={
    2017: "IDEB_AF_2017",
    2019: "IDEB_AF_2019"
})

df_af_delta["delta_IDEB_AF"] = (
    df_af_delta["IDEB_AF_2019"] - df_af_delta["IDEB_AF_2017"]
)

# tratamento em 2017
df_trat_2017 = (
    df[df["ANO"] == 2017][
        [
            "CO_ENTIDADE",
            "tratamento_real",
            "pct_bolsa_familia",
            "corte_50",
            "QT_MAT_FUND",
            "urbana",
            "dep_estadual",
            "dep_municipal"
        ]
    ]
    .copy()
    .rename(columns={
        "tratamento_real": "tratamento_2017",
        "pct_bolsa_familia": "pct_bolsa_familia_2017",
        "corte_50": "corte_50_2017",
        "QT_MAT_FUND": "QT_MAT_FUND_2017"
    })
)

# juntar deltas
df_delta = df_trat_2017.merge(
    df_ai_delta,
    on="CO_ENTIDADE",
    how="left"
)

df_delta = df_delta.merge(
    df_af_delta,
    on="CO_ENTIDADE",
    how="left"
)

# -------------------------------
# 8. EDA dos deltas
# -------------------------------

print("\n==============================")
print("17. BASE DELTA 2017-2019")
print("==============================")
print(df_delta.shape)
print(df_delta.head())

print("\n==============================")
print("18. RESUMO DELTA IDEB_AI")
print("==============================")
print(df_delta["delta_IDEB_AI"].describe())

print("\n==============================")
print("19. RESUMO DELTA IDEB_AF")
print("==============================")
print(df_delta["delta_IDEB_AF"].describe())

print("\n==============================")
print("20. DELTA IDEB_AI POR tratamento_2017")
print("==============================")
print(df_delta.groupby("tratamento_2017")["delta_IDEB_AI"].describe())

print("\n==============================")
print("21. DELTA IDEB_AF POR tratamento_2017")
print("==============================")
print(df_delta.groupby("tratamento_2017")["delta_IDEB_AF"].describe())

print("\n==============================")
print("22. MÉDIA DELTA IDEB_AI POR tratamento_2017")
print("==============================")
print(df_delta.groupby("tratamento_2017")["delta_IDEB_AI"].mean())

print("\n==============================")
print("23. MÉDIA DELTA IDEB_AF POR tratamento_2017")
print("==============================")
print(df_delta.groupby("tratamento_2017")["delta_IDEB_AF"].mean())

print("\n==============================")
print("24. DELTA IDEB_AI POR corte_50_2017")
print("==============================")
print(df_delta.groupby("corte_50_2017")["delta_IDEB_AI"].describe())

print("\n==============================")
print("25. DELTA IDEB_AF POR corte_50_2017")
print("==============================")
print(df_delta.groupby("corte_50_2017")["delta_IDEB_AF"].describe())

# -------------------------------
# 9. Salvar base delta
# -------------------------------

output_delta = r"C:\Users\suporte1\Desktop\lai\base_delta_ideb_2017_2019.csv"

df_delta.to_csv(output_delta, index=False, encoding="utf-8-sig")

print("\n==============================")
print("26. ARQUIVO DELTA SALVO")
print("==============================")
print(output_delta)
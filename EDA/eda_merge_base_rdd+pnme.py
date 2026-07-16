import pandas as pd

# -------------------------------
# 1. Caminhos
# -------------------------------

path_rdd = r"C:\Users\suporte1\Desktop\lai\base_rdd_escolas_2016_2019.csv"

path_pnme = r"C:\Users\suporte1\Desktop\lai\script_pnme\outputs\1.bases_iniciais\escolas_pnme_dedup_co_escola_ano.csv"

output_path = r"C:\Users\suporte1\Desktop\lai\base_rdd_pnme_2016_2019_defasado_corte_040_060.csv"
# -------------------------------
# 2. Ler bases
# -------------------------------

df_rdd = pd.read_csv(path_rdd, dtype={"CO_ENTIDADE": str})

df_pnme = pd.read_csv(
    path_pnme,
    sep=";",
    dtype=str,
)

df_merge = pd.read_csv(output_path, dtype={"CO_ENTIDADE": str})

print("\nBase RDD:")
print(df_rdd.shape)
print(df_rdd.head())

print("\nBase PNME:")
print(df_pnme.shape)
print(df_pnme.head())

print("\nColunas RDD:")
print(df_rdd.columns.tolist())

print("\nColunas PNME:")
print(df_pnme.columns.tolist())

print("\nMERGE:")
print(df_merge.shape)
print(df_merge.head())

print("\nColunas MERGE:")
print(df_merge.columns.tolist())

df_merge["ANO"] = df_merge["ANO"].astype(int)
df_merge["tratamento_real"] = df_merge["tratamento_real"].astype(int)
df_merge["corte_50"] = df_merge["corte_50"].astype(int)
df_merge["pct_bolsa_familia"] = df_merge["pct_bolsa_familia"].astype(float)

print("\n==============================")
print("1. DIMENSÃO")
print("==============================")
print(df_merge.shape)

print("\n==============================")
print("2. DUPLICATAS CO_ENTIDADE + ANO")
print("==============================")
print(df_merge.duplicated(["CO_ENTIDADE", "ANO"]).sum())

print("\n==============================")
print("3. DISTRIBUIÇÃO tratamento_real")
print("==============================")
print(df_merge["tratamento_real"].value_counts(dropna=False))

print("\n==============================")
print("4. TRATAMENTO POR ANO")
print("==============================")
print(pd.crosstab(df_merge["ANO"], df_merge["tratamento_real"]))

print("\n==============================")
print("5. PERCENTUAL TRATADO POR ANO")
print("==============================")
print(df_merge.groupby("ANO")["tratamento_real"].mean())

print("\n==============================")
print("6. PERCENTUAL TRATADO POR corte_50")
print("==============================")
print(df_merge.groupby("corte_50")["tratamento_real"].mean())

print("\n==============================")
print("7. TRATAMENTO POR ANO E corte_50")
print("==============================")
print(df_merge.groupby(["ANO", "corte_50"])["tratamento_real"].mean())

print("\n==============================")
print("8. CROSSTAB corte_50 x tratamento_real")
print("==============================")
print(pd.crosstab(df_merge["corte_50"], df_merge["tratamento_real"]))

print("\n==============================")
print("9. CROSSTAB percentual corte_50 x tratamento_real")
print("==============================")
print(
    pd.crosstab(
        df_merge["corte_50"],
        df_merge["tratamento_real"],
        normalize="index"
    )
)

print("\n==============================")
print("10. MISSING NAS COLUNAS PNME ENTRE TRATADOS")
print("==============================")

cols_pnme = [
    "pagina",
    "sg_uf",
    "no_municipio",
    "cnpj_executora",
    "executora",
    "sg_destinacao",
    "no_escola",
    "qt_alunos",
    "vl_custeio",
    "vl_capital",
    "vl_total",
    "esfera",
    "localizacao"
]

print(
    df_merge.loc[df_merge["tratamento_real"] == 1, cols_pnme]
    .isna()
    .sum()
)

print("\n==============================")
print("11. RESUMO pct_bolsa_familia")
print("==============================")
print(df_merge["pct_bolsa_familia"].describe())

print("\n==============================")
print("12. pct_bolsa_familia POR tratamento_real")
print("==============================")
print(df_merge.groupby("tratamento_real")["pct_bolsa_familia"].describe())

print("\n==============================")
print("13. pct_bolsa_familia POR corte_50")
print("==============================")
print(df_merge.groupby("corte_50")["pct_bolsa_familia"].describe())

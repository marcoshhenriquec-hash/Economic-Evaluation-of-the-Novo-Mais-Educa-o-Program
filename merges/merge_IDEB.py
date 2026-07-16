import pandas as pd

# -------------------------------
# Caminhos
# -------------------------------

path_base = r"C:\Users\suporte1\Desktop\lai\base_rdd_escolas_2016_2019.csv"
path_ideb = r"C:\Users\suporte1\Desktop\lai\script_pnme\outputs\4.tratamento_IDEB\ideb_2015_2017_2019_wide.csv"

output_merge_ano = r"C:\Users\suporte1\Desktop\lai\base_rdd_pnme_ideb_2015_2017_2019_por_ano.csv"
output_merge_escola = r"C:\Users\suporte1\Desktop\lai\base_modelo_ideb_2015_2019.csv"

# -------------------------------
# Ler bases
# -------------------------------

df_base = pd.read_csv(path_base, dtype={"CO_ENTIDADE": str})
df_ideb = pd.read_csv(path_ideb, dtype={"CO_ENTIDADE": str})

df_base.columns = df_base.columns.str.strip()
df_ideb.columns = df_ideb.columns.str.strip()

df_base["ANO"] = df_base["ANO"].astype(int)
df_ideb["ANO"] = df_ideb["ANO"].astype(int)

df_base["CO_ENTIDADE"] = (
    df_base["CO_ENTIDADE"]
    .astype(str)
    .str.strip()
    .str.zfill(8)
)

df_ideb["CO_ENTIDADE"] = (
    df_ideb["CO_ENTIDADE"]
    .astype(str)
    .str.strip()
    .str.zfill(8)
)

# -------------------------------
# 1) Merge por escola-ano
# -------------------------------

df_merge_ano = df_base.merge(
    df_ideb,
    on=["CO_ENTIDADE", "ANO"],
    how="left"
)

print("\n==============================")
print("MERGE POR ESCOLA-ANO")
print("==============================")

print("\nBase original:")
print(df_base.shape)

print("\nBase IDEB:")
print(df_ideb.shape)

print("\nBase com IDEB:")
print(df_merge_ano.shape)

print("\nDuplicatas CO_ENTIDADE + ANO:")
print(df_merge_ano.duplicated(["CO_ENTIDADE", "ANO"]).sum())

print("\nMissing IDEB por ano:")
print(
    df_merge_ano
    .groupby("ANO")[["IDEB_AI", "IDEB_AF"]]
    .apply(lambda x: x.isna().mean())
)

print("\nQuantidade com IDEB por ano:")
print(
    df_merge_ano
    .groupby("ANO")[["IDEB_AI", "IDEB_AF"]]
    .apply(lambda x: x.notna().sum())
)

print("\nTratamento entre escolas com IDEB_AI:")
print(
    df_merge_ano[df_merge_ano["IDEB_AI"].notna()]
    .groupby("ANO")["tratamento_real"]
    .mean()
)

print("\nTratamento entre escolas com IDEB_AF:")
print(
    df_merge_ano[df_merge_ano["IDEB_AF"].notna()]
    .groupby("ANO")["tratamento_real"]
    .mean()
)

df_merge_ano.to_csv(output_merge_ano, index=False, encoding="utf-8-sig")

print("\nArquivo por escola-ano salvo em:")
print(output_merge_ano)

# -------------------------------
# 2) Transformar IDEB em base por escola
# -------------------------------

df_ideb_escola = (
    df_ideb
    .pivot_table(
        index="CO_ENTIDADE",
        columns="ANO",
        values=["IDEB_AI", "IDEB_AF"],
        aggfunc="first"
    )
)

df_ideb_escola.columns = [
    f"{var}_{ano}" for var, ano in df_ideb_escola.columns
]

df_ideb_escola = df_ideb_escola.reset_index()

# -------------------------------
# 3) Pegar somente linha de 2017 da base principal
# -------------------------------

df_2017 = df_base[df_base["ANO"] == 2017].copy()

# Renomear variáveis de controle para deixar claro que são de 2017
cols_renomear = {
    "QT_MAT_FUND": "QT_MAT_FUND_2017",
    "QT_BENEFICIARIOS": "QT_BENEFICIARIOS_2017",
    "pct_bolsa_familia": "pct_bolsa_familia_2017",
    "corte_50": "corte_50_2017",
    "tratamento_real": "tratamento_2017"
}

df_2017 = df_2017.rename(columns=cols_renomear)

# -------------------------------
# 4) Merge escola 2017 + IDEB 2015/2017/2019
# -------------------------------

df_modelo = df_2017.merge(
    df_ideb_escola,
    on="CO_ENTIDADE",
    how="left"
)

# -------------------------------
# 5) Criar deltas IDEB
# -------------------------------

df_modelo["delta_IDEB_AI_2015_2019"] = (
    df_modelo["IDEB_AI_2019"] - df_modelo["IDEB_AI_2015"]
)

df_modelo["delta_IDEB_AF_2015_2019"] = (
    df_modelo["IDEB_AF_2019"] - df_modelo["IDEB_AF_2015"]
)

df_modelo["delta_IDEB_AI_2017_2019"] = (
    df_modelo["IDEB_AI_2019"] - df_modelo["IDEB_AI_2017"]
)

df_modelo["delta_IDEB_AF_2017_2019"] = (
    df_modelo["IDEB_AF_2019"] - df_modelo["IDEB_AF_2017"]
)

# -------------------------------
# Checks da base de modelo
# -------------------------------

print("\n==============================")
print("BASE DE MODELO POR ESCOLA")
print("==============================")

print("\nBase 2017:")
print(df_2017.shape)

print("\nIDEB por escola:")
print(df_ideb_escola.shape)

print("\nBase modelo:")
print(df_modelo.shape)

print("\nDuplicatas CO_ENTIDADE:")
print(df_modelo.duplicated("CO_ENTIDADE").sum())

print("\nColunas IDEB disponíveis:")
cols_ideb = [
    "IDEB_AI_2015", "IDEB_AI_2017", "IDEB_AI_2019",
    "IDEB_AF_2015", "IDEB_AF_2017", "IDEB_AF_2019",
    "delta_IDEB_AI_2015_2019", "delta_IDEB_AF_2015_2019",
    "delta_IDEB_AI_2017_2019", "delta_IDEB_AF_2017_2019"
]

print(df_modelo[cols_ideb].notna().sum())

print("\nTratamento 2017:")
print(df_modelo["tratamento_2017"].value_counts(dropna=False))

print("\nTratamento 2017 entre escolas com IDEB_AI_2015 e IDEB_AI_2019:")
print(
    df_modelo[
        df_modelo["IDEB_AI_2015"].notna() &
        df_modelo["IDEB_AI_2019"].notna()
    ]["tratamento_2017"].mean()
)

print("\nTratamento 2017 entre escolas com IDEB_AF_2015 e IDEB_AF_2019:")
print(
    df_modelo[
        df_modelo["IDEB_AF_2015"].notna() &
        df_modelo["IDEB_AF_2019"].notna()
    ]["tratamento_2017"].mean()
)

print("\nResumo dos deltas:")
print(
    df_modelo[
        ["delta_IDEB_AI_2015_2019", "delta_IDEB_AF_2015_2019"]
    ].describe()
)

# -------------------------------
# Salvar
# -------------------------------

df_modelo.to_csv(output_merge_escola, index=False, encoding="utf-8-sig")

print("\nArquivo de modelo salvo em:")
print(output_merge_escola)
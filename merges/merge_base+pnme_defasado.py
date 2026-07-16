import pandas as pd

# -------------------------------
# 1. Caminhos
# -------------------------------

path_rdd = r"C:\Users\suporte1\Desktop\lai\base_rdd_escolas_2016_2019.csv"

path_pnme = r"C:\Users\suporte1\Desktop\lai\script_pnme\outputs\1.bases_iniciais\escolas_pnme_dedup_co_escola_ano.csv"

output_path = r"C:\Users\suporte1\Desktop\lai\base_rdd_pnme_2016_2019_defasado.csv"

# -------------------------------
# 2. Ler bases
# -------------------------------

df_rdd = pd.read_csv(
    path_rdd,
    dtype={"CO_ENTIDADE": str}
)

df_pnme = pd.read_csv(
    path_pnme,
    sep=";",
    dtype={"CO_ENTIDADE": str}
)

# -------------------------------
# 3. Padronizar anos
# -------------------------------

df_rdd["ANO"] = pd.to_numeric(df_rdd["ANO"], errors="raise").astype(int)
df_pnme["ANO"] = pd.to_numeric(df_pnme["ANO"], errors="raise").astype(int)

# A base explicativa de 2016 é associada ao PNME de 2017;
# a de 2017 ao PNME de 2018; e a de 2018 ao PNME de 2019.
df_rdd = df_rdd[df_rdd["ANO"].between(2016, 2018)].copy()
df_rdd["ANO_PNME"] = df_rdd["ANO"] + 1

df_pnme = (
    df_pnme[df_pnme["ANO"].between(2017, 2019)]
    .rename(columns={"ANO": "ANO_PNME"})
    .copy()
)

df_pnme["tratamento_real"] = 1

# -------------------------------
# 4. Checar unicidade das chaves
# -------------------------------

chaves = ["CO_ENTIDADE", "ANO_PNME"]

if df_rdd.duplicated(chaves).any():
    raise ValueError(
        "A base RDD possui duplicatas por CO_ENTIDADE e ANO_PNME. "
        "O merge poderia multiplicar observações."
    )

if df_pnme.duplicated(chaves).any():
    raise ValueError(
        "A base PNME possui duplicatas por CO_ENTIDADE e ANO_PNME. "
        "Revise a deduplicação antes do merge."
    )

# -------------------------------
# 5. Merge com defasagem de um ano
# -------------------------------

df_final = df_rdd.merge(
    df_pnme,
    on=chaves,
    how="left",
    validate="one_to_one",
    suffixes=("", "_PNME")
)

# Escolas não encontradas na lista do PNME naquele ano são controles.
df_final["tratamento_real"] = (
    df_final["tratamento_real"]
    .fillna(0)
    .astype(int)
)

# -------------------------------
# 6. Salvar e conferir
# -------------------------------

df_final.to_csv(output_path, index=False)

print("Merge com defasagem concluído.")
print("Base final:", df_final.shape)
print("\nAssociação utilizada:")
print(
    df_final[["ANO", "ANO_PNME"]]
    .drop_duplicates()
    .sort_values("ANO")
    .to_string(index=False)
)
print("\nTratamento real por ano do PNME:")
print(
    df_final.groupby("ANO_PNME")["tratamento_real"]
    .agg(observacoes="size", tratadas="sum")
    .to_string()
)
print("\nArquivo salvo em:")
print(output_path)
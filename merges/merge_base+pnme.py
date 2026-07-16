import pandas as pd

# -------------------------------
# 1. Caminhos
# -------------------------------

path_rdd = r"C:\Users\suporte1\Desktop\lai\base_rdd_escolas_2016_2019.csv"

path_pnme = r"C:\Users\suporte1\Desktop\lai\script_pnme\outputs\1.bases_iniciais\escolas_pnme_dedup_co_escola_ano.csv"

output_path = r"C:\Users\suporte1\Desktop\lai\base_rdd_pnme_2016_2019.csv"

# -------------------------------
# 2. Ler bases
# -------------------------------

df_rdd = pd.read_csv(path_rdd, dtype={"CO_ENTIDADE": str})

df_pnme = pd.read_csv(
    path_pnme,
    sep=";",
    dtype=str
)

df_rdd["ANO"] = df_rdd["ANO"].astype(int)
df_pnme["ANO"] = df_pnme["ANO"].astype(int)

df_pnme["tratamento_real"] = 1

# -------------------------------
# 5. Merge
# -------------------------------

df_final = df_rdd.merge(
    df_pnme,
    on=["CO_ENTIDADE", "ANO"],
    how="left"
)

df_final["tratamento_real"] = (
    df_final["tratamento_real"]
    .fillna(0)
    .astype(int)
)

# -------------------------------
# 6. Salvar
# -------------------------------

df_final.to_csv(output_path, index=False)

print("Merge concluído.")
print("Base final:", df_final.shape)
print("Arquivo salvo em:")
print(output_path)
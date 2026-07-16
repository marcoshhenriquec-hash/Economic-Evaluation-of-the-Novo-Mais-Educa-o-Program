import pandas as pd

# -------------------------------
# Caminhos
# -------------------------------

path_ai = r"C:\Users\suporte1\Desktop\lai\IDEB\ideb_anos_iniciais.csv"
path_af = r"C:\Users\suporte1\Desktop\lai\IDEB\ideb_anos_finais.csv"

output_long = r"C:\Users\suporte1\Desktop\lai\IDEB\ideb_2015_2017_2019_long.csv"
output_wide = r"C:\Users\suporte1\Desktop\lai\IDEB\ideb_2015_2017_2019_wide.csv"

# -------------------------------
# Ler CSVs pulando linhas superiores
# -------------------------------

df_ai = pd.read_csv(
    path_ai,
    sep=None,
    engine="python",
    dtype=str,
    encoding="utf-8-sig",
    skiprows=31,
    header=None,
    on_bad_lines="skip"
)

df_af = pd.read_csv(
    path_af,
    sep=None,
    engine="python",
    dtype=str,
    encoding="utf-8-sig",
    skiprows=31,
    header=None,
    on_bad_lines="skip"
)

print("\nAI shape:")
print(df_ai.shape)

print("\nAF shape:")
print(df_af.shape)

# -------------------------------
# Conferência das colunas relevantes
# -------------------------------

print("\nAI head:")
print(df_ai.iloc[:5, [0, 1, 2, 3, 4, 5, 111, 112, 113]].to_string())

print("\nAF head:")
print(df_af.iloc[:5, [0, 1, 2, 3, 4, 5, 101, 102, 103]].to_string())

# -------------------------------
# Função de tratamento por posição
# -------------------------------

def tratar_ideb(df, etapa, col_ideb_2015, col_ideb_2017, col_ideb_2019):

    df_limpo = pd.DataFrame()

    df_limpo["CO_ENTIDADE"] = df.iloc[:, 3]
    df_limpo["NO_ESCOLA"] = df.iloc[:, 4]

    df_limpo["IDEB_2015"] = df.iloc[:, col_ideb_2015]
    df_limpo["IDEB_2017"] = df.iloc[:, col_ideb_2017]
    df_limpo["IDEB_2019"] = df.iloc[:, col_ideb_2019]

    df_limpo["etapa"] = etapa

    df_limpo["CO_ENTIDADE"] = (
        df_limpo["CO_ENTIDADE"]
        .astype(str)
        .str.strip()
        .str.zfill(8)
    )

    df_long = df_limpo.melt(
        id_vars=["CO_ENTIDADE", "NO_ESCOLA", "etapa"],
        value_vars=["IDEB_2015", "IDEB_2017", "IDEB_2019"],
        var_name="ano_ideb",
        value_name="IDEB"
    )

    df_long["ANO"] = (
        df_long["ano_ideb"]
        .str.extract(r"(\d{4})")
        .astype(int)
    )

    df_long["IDEB"] = (
        df_long["IDEB"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("-", "", regex=False)
        .str.strip()
    )

    df_long["IDEB"] = pd.to_numeric(df_long["IDEB"], errors="coerce")

    df_long = df_long.dropna(subset=["IDEB"]).copy()

    df_long = df_long[
        ["CO_ENTIDADE", "NO_ESCOLA", "ANO", "etapa", "IDEB"]
    ].copy()

    return df_long

# -------------------------------
# Tratar AI e AF
# -------------------------------

df_ai_long = tratar_ideb(
    df_ai,
    etapa="AI",
    col_ideb_2015=111,
    col_ideb_2017=112,
    col_ideb_2019=113
)

df_af_long = tratar_ideb(
    df_af,
    etapa="AF",
    col_ideb_2015=101,
    col_ideb_2017=102,
    col_ideb_2019=103
)

# -------------------------------
# Juntar
# -------------------------------

df_long = pd.concat([df_ai_long, df_af_long], ignore_index=True)

# -------------------------------
# Wide para merge
# -------------------------------

df_wide = (
    df_long
    .pivot_table(
        index=["CO_ENTIDADE", "ANO"],
        columns="etapa",
        values="IDEB",
        aggfunc="first"
    )
    .reset_index()
)

df_wide = df_wide.rename(columns={
    "AI": "IDEB_AI",
    "AF": "IDEB_AF"
})

# -------------------------------
# Checks
# -------------------------------

print("\n==============================")
print("IDEB LONG")
print("==============================")
print(df_long.shape)
print(df_long.head())

print("\nEscolas únicas por ANO e etapa:")
print(df_long.groupby(["ANO", "etapa"])["CO_ENTIDADE"].nunique())

print("\nResumo IDEB por ANO e etapa:")
print(df_long.groupby(["ANO", "etapa"])["IDEB"].describe())

print("\n==============================")
print("IDEB WIDE")
print("==============================")
print(df_wide.shape)
print(df_wide.head())

print("\nAnos no wide:")
print(df_wide["ANO"].value_counts().sort_index())

print("\nDuplicatas CO_ENTIDADE + ANO no wide:")
print(df_wide.duplicated(["CO_ENTIDADE", "ANO"]).sum())

print("\nMissing IDEB_AI e IDEB_AF:")
print(df_wide[["IDEB_AI", "IDEB_AF"]].isna().sum())

# -------------------------------
# Salvar
# -------------------------------

df_long.to_csv(output_long, index=False, encoding="utf-8-sig")
df_wide.to_csv(output_wide, index=False, encoding="utf-8-sig")

print("\nArquivos salvos:")
print(output_long)
print(output_wide)
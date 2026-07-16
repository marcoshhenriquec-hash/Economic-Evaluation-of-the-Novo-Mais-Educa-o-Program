# ======================================================
# 02_merge_censo_bf.py
# Merge Censo Escolar 2016-2019 + Bolsa Família por série
# ======================================================

from pathlib import Path

import pandas as pd


# ======================================================
# 1. Caminhos
# ======================================================

path_censo = Path(
    r"C:\Users\suporte1\Desktop\lai\censo\censo_escolar_2016_2019.csv"
)

path_bf = Path(
    r"C:\Users\suporte1\Desktop\lai\2017_a_2019_pbf\2016_2017_a_2019_pbf_concat.csv"
)

path_saida = Path(
    r"C:\Users\suporte1\Desktop\lai\base_rdd_escolas_2016_2019.csv"
)


# ======================================================
# 2. Função de leitura da base BF
# ======================================================

def ler_bf(path: Path) -> pd.DataFrame:
    """
    Tenta ler o arquivo com os encodings mais prováveis.
    """
    erros = []

    for encoding in ("utf-8-sig", "latin1"):
        try:
            return pd.read_csv(
                path,
                sep=";",
                encoding=encoding,
                dtype={"INEP_ESCOLA": "string"},
                low_memory=False,
            )
        except UnicodeDecodeError as exc:
            erros.append(f"{encoding}: {exc}")

    raise UnicodeError(
        "Não foi possível ler a base BF.\n" + "\n".join(erros)
    )


# ======================================================
# 3. Validar arquivos
# ======================================================

if not path_censo.exists():
    raise FileNotFoundError(f"Base do Censo não encontrada: {path_censo}")

if not path_bf.exists():
    raise FileNotFoundError(f"Base do Bolsa Família não encontrada: {path_bf}")


# ======================================================
# 4. Carregar Censo Escolar
# ======================================================

print(f"Lendo Censo: {path_censo}")

df_censo = pd.read_csv(
    path_censo,
    dtype={"CO_ENTIDADE": "string"},
    low_memory=False,
)

df_censo = df_censo.rename(columns={"NU_ANO_CENSO": "ANO"})

df_censo["CO_ENTIDADE"] = (
    df_censo["CO_ENTIDADE"]
    .astype("string")
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(8)
)

df_censo["ANO"] = pd.to_numeric(
    df_censo["ANO"],
    errors="coerce",
).astype("Int64")

df_censo = df_censo[
    df_censo["ANO"].between(2016, 2019)
].copy()

print("Censo carregado:", df_censo.shape)
print("Censo por ano:")
print(df_censo["ANO"].value_counts().sort_index())


# ======================================================
# 5. Carregar base BF concatenada
# ======================================================

print(f"\nLendo Bolsa Família: {path_bf}")

df_bf = ler_bf(path_bf)

# Limpar possíveis caracteres invisíveis nos nomes das colunas
df_bf.columns = (
    df_bf.columns
    .astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

if "INEP_ESCOLA" not in df_bf.columns:
    raise KeyError(
        "A coluna INEP_ESCOLA não foi encontrada na base BF. "
        f"Colunas disponíveis: {df_bf.columns.tolist()}"
    )

if "ANO" not in df_bf.columns:
    raise KeyError(
        "A coluna ANO não foi encontrada na base BF. "
        f"Colunas disponíveis: {df_bf.columns.tolist()}"
    )

if "SERIE" not in df_bf.columns:
    raise KeyError(
        "A coluna SERIE não foi encontrada na base BF. "
        f"Colunas disponíveis: {df_bf.columns.tolist()}"
    )

df_bf = df_bf.rename(columns={"INEP_ESCOLA": "CO_ENTIDADE"})

df_bf["CO_ENTIDADE"] = (
    df_bf["CO_ENTIDADE"]
    .astype("string")
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(8)
)

df_bf["ANO"] = pd.to_numeric(
    df_bf["ANO"],
    errors="coerce",
).astype("Int64")

df_bf = df_bf[
    df_bf["ANO"].between(2016, 2019)
].copy()

print("BF carregado:", df_bf.shape)
print("BF por ano:")
print(df_bf["ANO"].value_counts().sort_index())


# ======================================================
# 6. Construir beneficiários por série
# ======================================================

cols_meses = [
    "JANAIRO",
    "FEVEREIRO",
    "MARCO",
    "ABRIL",
    "MAIO",
    "JUNHO",
    "JULHO",
    "AGOSTO",
    "SETEMBRO",
    "OUTUBRO",
    "NOVEMBRO",
    "DEZEMBRO",
]

meses_ausentes = [
    coluna for coluna in cols_meses
    if coluna not in df_bf.columns
]

if meses_ausentes:
    raise KeyError(
        "Meses ausentes na base BF: "
        + ", ".join(meses_ausentes)
    )

# Garantir que os meses sejam numéricos
for coluna in cols_meses:
    df_bf[coluna] = pd.to_numeric(
        df_bf[coluna],
        errors="coerce",
    )

# Para cada escola-série-ano, usar o maior valor mensal
df_bf["BF_SERIE"] = df_bf[cols_meses].max(
    axis=1,
    skipna=True,
)


# ======================================================
# 7. Filtrar séries do ensino fundamental
# ======================================================

serie_normalizada = (
    df_bf["SERIE"]
    .astype("string")
    .str.normalize("NFKD")
    .str.encode("ascii", errors="ignore")
    .str.decode("utf-8")
    .str.lower()
    .str.strip()
)

df_bf = df_bf[
    serie_normalizada.str.contains(
        "ensino fundamental",
        na=False,
    )
].copy()

serie_normalizada = (
    df_bf["SERIE"]
    .astype("string")
    .str.normalize("NFKD")
    .str.encode("ascii", errors="ignore")
    .str.decode("utf-8")
    .str.lower()
    .str.strip()
)

df_bf = df_bf[
    ~serie_normalizada.str.contains(
        "nao informada",
        na=False,
    )
].copy()

print("\nBF após filtro do ensino fundamental:", df_bf.shape)
print("Linhas BF por ano após o filtro:")
print(df_bf["ANO"].value_counts().sort_index())


# ======================================================
# 8. Agregar beneficiários por escola-ano
# ======================================================

df_bf_ano = (
    df_bf
    .groupby(
        ["CO_ENTIDADE", "ANO"],
        as_index=False,
        dropna=False,
    )["BF_SERIE"]
    .sum(min_count=1)
    .rename(columns={"BF_SERIE": "QT_BENEFICIARIOS"})
)

df_bf_ano["QT_BENEFICIARIOS"] = (
    df_bf_ano["QT_BENEFICIARIOS"]
    .fillna(0)
)

print("\nBase BF agregada:", df_bf_ano.shape)
print("Escolas-ano BF por ano:")
print(df_bf_ano["ANO"].value_counts().sort_index())


# ======================================================
# 9. Merge Censo + BF
# ======================================================

df = df_censo.merge(
    df_bf_ano,
    on=["CO_ENTIDADE", "ANO"],
    how="left",
    validate="one_to_one",
)

# Manter todas as escolas do Censo
df["QT_BENEFICIARIOS"] = (
    df["QT_BENEFICIARIOS"]
    .fillna(0)
)


# ======================================================
# 10. Construir running variable
# ======================================================

df["pct_bolsa_familia"] = (
    df["QT_BENEFICIARIOS"]
    / df["QT_MAT_FUND"]
)


# ======================================================
# 11. Limpeza lógica
# ======================================================

violacoes_antes = (
    df["QT_BENEFICIARIOS"]
    > df["QT_MAT_FUND"]
).sum()

print(
    "\nViolações antes do filtro "
    "(beneficiários > matrículas):",
    violacoes_antes,
)

df = df[
    df["pct_bolsa_familia"].between(0, 1)
].copy()


# ======================================================
# 12. Indicador do corte de 50%
# ======================================================

df["corte_50"] = (
    df["pct_bolsa_familia"] >= 0.5
).astype(int)


# ======================================================
# 13. Salvar base final
# ======================================================

df.to_csv(
    path_saida,
    index=False,
    encoding="utf-8-sig",
)

print(f"\nBase salva em: {path_saida}")


# ======================================================
# 14. Sanity checks
# ======================================================

print("\n==============================")
print("SANITY CHECKS")
print("==============================")

print("\n1. Duplicatas BF escola-ano:")
print(
    df_bf_ano
    .duplicated(["CO_ENTIDADE", "ANO"])
    .sum()
)

print("\n2. Duplicatas finais escola-ano:")
print(
    df
    .duplicated(["CO_ENTIDADE", "ANO"])
    .sum()
)

print("\n3. Tamanho das bases:")
print("Censo:", len(df_censo))
print("BF agregado:", len(df_bf_ano))
print("Final:", len(df))

print("\n4. Observações finais por ano:")
print(df["ANO"].value_counts().sort_index())

print("\n5. Missing BF:")
print(df["QT_BENEFICIARIOS"].isna().sum())

print("\n6. Violações lógicas após o filtro:")
print(
    (
        df["QT_BENEFICIARIOS"]
        > df["QT_MAT_FUND"]
    ).sum()
)

print("\n7. Estatísticas básicas:")
print(
    df[
        [
            "QT_MAT_FUND",
            "QT_BENEFICIARIOS",
        ]
    ].describe()
)

print("\n8. Running variable:")
print(df["pct_bolsa_familia"].describe())

print("\n9. Running variable por ano:")
print(
    df.groupby("ANO")["pct_bolsa_familia"]
    .describe()
)

print("\n10. Percentual acima do corte por ano:")
print(
    df.groupby("ANO")["corte_50"]
    .mean()
)
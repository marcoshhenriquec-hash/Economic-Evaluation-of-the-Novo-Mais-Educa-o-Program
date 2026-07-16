from pathlib import Path

import pandas as pd

# Pasta onde estão os arquivos:
# microdados_ed_basica_2016.csv
# microdados_ed_basica_2017.csv
# microdados_ed_basica_2018.csv
# microdados_ed_basica_2019.csv
path_censo = Path(r"C:\Users\suporte1\Desktop\lai\censo")

# Arquivo de saída
path_saida = path_censo / "censo_escolar_2016_2019.csv"

# Colunas relevantes
cols_censo = [
    "NU_ANO_CENSO",
    "CO_ENTIDADE",
    "QT_MAT_FUND",
    "TP_LOCALIZACAO",
    "TP_DEPENDENCIA",
]

# Período atualizado: 2016 a 2019
anos = range(2016, 2020)

dfs = []

for ano in anos:
    path = path_censo / f"microdados_ed_basica_{ano}.csv"

    print(f"Lendo Censo {ano}: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df_ano = pd.read_csv(
        path,
        sep=";",
        encoding="latin1",
        usecols=cols_censo,
        dtype={"CO_ENTIDADE": "string"},
        low_memory=False,
    )

    # -------------------------------
    # filtros estruturais
    # -------------------------------

    # Só escolas com matrículas no ensino fundamental
    df_ano = df_ano[df_ano["QT_MAT_FUND"] > 0].copy()

    # Excluir rede privada
    df_ano = df_ano[df_ano["TP_DEPENDENCIA"] != 4].copy()

    # -------------------------------
    # dummies
    # -------------------------------

    df_ano["urbana"] = (df_ano["TP_LOCALIZACAO"] == 1).astype(int)
    df_ano["dep_estadual"] = (df_ano["TP_DEPENDENCIA"] == 2).astype(int)
    df_ano["dep_municipal"] = (df_ano["TP_DEPENDENCIA"] == 3).astype(int)

    # Manter somente as colunas necessárias
    df_ano = df_ano[
        [
            "NU_ANO_CENSO",
            "CO_ENTIDADE",
            "QT_MAT_FUND",
            "urbana",
            "dep_estadual",
            "dep_municipal",
        ]
    ].copy()

    dfs.append(df_ano)
    del df_ano

# -------------------------------
# concatenar
# -------------------------------

df_censo = pd.concat(dfs, ignore_index=True)

print("Todos os anos concatenados")

# -------------------------------
# garantir unicidade escola-ano
# -------------------------------

duplicados = df_censo.duplicated(["CO_ENTIDADE", "NU_ANO_CENSO"]).sum()
print("Duplicatas escola-ano:", duplicados)

if duplicados > 0:
    df_censo = (
        df_censo.groupby(
            ["CO_ENTIDADE", "NU_ANO_CENSO"],
            as_index=False,
        )
        .agg(
            {
                "QT_MAT_FUND": "sum",
                "urbana": "max",
                "dep_estadual": "max",
                "dep_municipal": "max",
            }
        )
    )

# -------------------------------
# sanity checks
# -------------------------------

print("\nObservações por ano:")
print(df_censo["NU_ANO_CENSO"].value_counts().sort_index())

print("\nResumo matrículas:")
print(df_censo["QT_MAT_FUND"].describe())

print("\nDimensão final:", df_censo.shape)

# -------------------------------
# salvar
# -------------------------------

df_censo.to_csv(path_saida, index=False)

print(f"Arquivo salvo em: {path_saida}")

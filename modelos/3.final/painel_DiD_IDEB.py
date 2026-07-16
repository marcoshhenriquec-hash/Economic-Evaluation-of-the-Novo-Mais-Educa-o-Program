# ======================================================
# painel_DiD_IDEB.py
# Monta painel IDEB 2005-2019 + tratamento PNME 2017
# Versão robusta para arquivos IDEB 2023
# ======================================================

import pandas as pd
import numpy as np
import re
import unicodedata

# -------------------------------
# 1. Caminhos
# -------------------------------

path_ai = r"C:\Users\suporte1\Desktop\lai\IDEB\anos iniciais\divulgacao_anos_iniciais_escolas_2023\divulgacao_anos_iniciais_escolas_2023.csv"
path_af = r"C:\Users\suporte1\Desktop\lai\IDEB\anos finais\divulgacao_anos_finais_escolas_2023.csv"

path_base = r"C:\Users\suporte1\Desktop\lai\base_rdd_pnme_2017_2019.csv"

output_ideb_long = r"C:\Users\suporte1\Desktop\lai\IDEB\ideb_2005_2019_long.csv"
output_ideb_escola_ano = r"C:\Users\suporte1\Desktop\lai\IDEB\ideb_2005_2019_escola_ano.csv"
output_ideb_escola_wide = r"C:\Users\suporte1\Desktop\lai\IDEB\ideb_2005_2019_wide_escola.csv"
output_painel = r"C:\Users\suporte1\Desktop\lai\painel_ideb_pnme_2005_2019.csv"

anos_ideb = [2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019]


# -------------------------------
# 2. Funções auxiliares
# -------------------------------

def normalizar_coluna(x):
    x = str(x).strip()
    x = unicodedata.normalize("NFKD", x)
    x = "".join([c for c in x if not unicodedata.combining(c)])
    x = x.upper()
    x = re.sub(r"\s+", "_", x)
    x = re.sub(r"[^A-Z0-9_]", "_", x)
    x = re.sub(r"_+", "_", x)
    x = x.strip("_")
    return x


def ler_csv_ideb(path):
    """
    Lê o CSV tentando detectar o cabeçalho automaticamente.
    """
    # Primeiro lê tudo sem header para descobrir onde está o cabeçalho
    bruto = pd.read_csv(
        path,
        sep=None,
        engine="python",
        dtype=str,
        encoding="utf-8-sig",
        header=None,
        on_bad_lines="skip"
    )

    print("\nArquivo:", path)
    print("Shape bruto:", bruto.shape)

    header_row = None

    # Procura uma linha que pareça conter nomes de colunas
    for i in range(min(50, len(bruto))):
        linha = " ".join(bruto.iloc[i].dropna().astype(str).tolist()).upper()

        tem_escola = ("ESCOLA" in linha) or ("ENTIDADE" in linha) or ("INEP" in linha)
        tem_ideb = ("IDEB" in linha) or ("2005" in linha and "2019" in linha)

        if tem_escola and tem_ideb:
            header_row = i
            break

    if header_row is None:
        print("\nNão achei cabeçalho automaticamente. Vou tentar header=0.")
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            dtype=str,
            encoding="utf-8-sig",
            header=0,
            on_bad_lines="skip"
        )
    else:
        print("Cabeçalho detectado na linha:", header_row)
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            dtype=str,
            encoding="utf-8-sig",
            header=header_row,
            on_bad_lines="skip"
        )

    df.columns = [normalizar_coluna(c) for c in df.columns]

    # Remove linhas totalmente vazias
    df = df.dropna(how="all").copy()

    print("Shape tratado:", df.shape)
    print("\nPrimeiras colunas detectadas:")
    print(df.columns.tolist()[:40])

    return df


def encontrar_coluna_codigo(df):
    candidatos = [
        "CO_ENTIDADE",
        "ID_ESCOLA",
        "CO_ESCOLA",
        "INEP_ESCOLA",
        "COD_ESCOLA",
        "CODIGO_DA_ESCOLA",
        "CODIGO_ESCOLA"
    ]

    for c in candidatos:
        if c in df.columns:
            return c

    # busca flexível
    for c in df.columns:
        if ("ESCOLA" in c or "ENTIDADE" in c or "INEP" in c) and ("CO" in c or "ID" in c or "COD" in c):
            return c

    raise ValueError("Não encontrei coluna de código da escola. Veja as colunas impressas no output.")


def encontrar_coluna_nome(df):
    candidatos = [
        "NO_ESCOLA",
        "NOME_ESCOLA",
        "ESCOLA",
        "NOME_DA_ESCOLA"
    ]

    for c in candidatos:
        if c in df.columns:
            return c

    for c in df.columns:
        if "ESCOLA" in c and ("NO" in c or "NOME" in c):
            return c

    return None


def encontrar_colunas_ideb(df, anos):
    """
    Encontra as colunas de IDEB por ano.
    Procura nomes que tenham IDEB e o ano.
    Se não achar com IDEB, procura só pelo ano, mas com cuidado.
    """
    mapa = {}

    for ano in anos:
        ano_str = str(ano)

        candidatos_ideb = [
            c for c in df.columns
            if ano_str in c and "IDEB" in c
        ]

        if len(candidatos_ideb) == 1:
            mapa[ano] = candidatos_ideb[0]
            continue

        if len(candidatos_ideb) > 1:
            # tenta evitar meta/projeção se existir
            preferidos = [
                c for c in candidatos_ideb
                if "META" not in c and "PROJECAO" not in c and "PROJ" not in c
            ]

            if len(preferidos) >= 1:
                mapa[ano] = preferidos[0]
            else:
                mapa[ano] = candidatos_ideb[0]
            continue

        # fallback: colunas que contêm só o ano
        candidatos_ano = [
            c for c in df.columns
            if ano_str in c and "META" not in c and "PROJECAO" not in c and "PROJ" not in c
        ]

        if len(candidatos_ano) >= 1:
            mapa[ano] = candidatos_ano[0]
        else:
            print(f"Atenção: não encontrei coluna de IDEB para {ano}")

    return mapa


def tratar_ideb(df, etapa, anos):
    col_codigo = encontrar_coluna_codigo(df)
    col_nome = encontrar_coluna_nome(df)
    mapa_ideb = encontrar_colunas_ideb(df, anos)

    print("\n==============================")
    print(f"MAPEAMENTO {etapa}")
    print("==============================")
    print("Coluna código escola:", col_codigo)
    print("Coluna nome escola:", col_nome)
    print("Colunas IDEB encontradas:")
    for ano, col in mapa_ideb.items():
        print(ano, "->", col)

    base = pd.DataFrame()
    base["CO_ENTIDADE"] = df[col_codigo]

    if col_nome is not None:
        base["NO_ESCOLA"] = df[col_nome]
    else:
        base["NO_ESCOLA"] = np.nan

    base["etapa"] = etapa

    for ano, col in mapa_ideb.items():
        base[f"IDEB_{ano}"] = df[col]

    base["CO_ENTIDADE"] = (
        base["CO_ENTIDADE"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(8)
    )

    value_vars = [f"IDEB_{ano}" for ano in mapa_ideb.keys()]

    df_long = base.melt(
        id_vars=["CO_ENTIDADE", "NO_ESCOLA", "etapa"],
        value_vars=value_vars,
        var_name="ano_ideb",
        value_name="IDEB"
    )

    df_long["ANO"] = df_long["ano_ideb"].str.extract(r"(\d{4})").astype(int)

    df_long["IDEB"] = (
        df_long["IDEB"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("-", "", regex=False)
        .str.strip()
    )

    df_long["IDEB"] = pd.to_numeric(df_long["IDEB"], errors="coerce")

    df_long = df_long.dropna(subset=["IDEB"]).copy()
    df_long = df_long[df_long["IDEB"] > 0].copy()

    df_long = df_long[["CO_ENTIDADE", "NO_ESCOLA", "ANO", "etapa", "IDEB"]].copy()

    return df_long


# -------------------------------
# 3. Ler e tratar IDEB
# -------------------------------

df_ai_raw = ler_csv_ideb(path_ai)
df_af_raw = ler_csv_ideb(path_af)

df_ai_long = tratar_ideb(df_ai_raw, "AI", anos_ideb)
df_af_long = tratar_ideb(df_af_raw, "AF", anos_ideb)

df_ideb_long = pd.concat([df_ai_long, df_af_long], ignore_index=True)

print("\n==============================")
print("IDEB LONG 2005-2019")
print("==============================")
print(df_ideb_long.shape)
print(df_ideb_long.head())

print("\nEscolas únicas por ANO e etapa:")
print(df_ideb_long.groupby(["ANO", "etapa"])["CO_ENTIDADE"].nunique())

print("\nResumo IDEB por ANO e etapa:")
print(df_ideb_long.groupby(["ANO", "etapa"])["IDEB"].describe())


# -------------------------------
# 4. IDEB escola-ano em formato wide por etapa
# -------------------------------

df_ideb_escola_ano = (
    df_ideb_long
    .pivot_table(
        index=["CO_ENTIDADE", "ANO"],
        columns="etapa",
        values="IDEB",
        aggfunc="first"
    )
    .reset_index()
)

df_ideb_escola_ano = df_ideb_escola_ano.rename(columns={
    "AI": "IDEB_AI",
    "AF": "IDEB_AF"
})

print("\n==============================")
print("IDEB ESCOLA-ANO")
print("==============================")
print(df_ideb_escola_ano.shape)
print(df_ideb_escola_ano.head())

print("\nDuplicatas CO_ENTIDADE + ANO:")
print(df_ideb_escola_ano.duplicated(["CO_ENTIDADE", "ANO"]).sum())

print("\nAnos no IDEB escola-ano:")
print(df_ideb_escola_ano["ANO"].value_counts().sort_index())


# -------------------------------
# 5. IDEB wide por escola
# -------------------------------

df_ideb_escola_wide = (
    df_ideb_long
    .pivot_table(
        index="CO_ENTIDADE",
        columns=["etapa", "ANO"],
        values="IDEB",
        aggfunc="first"
    )
)

df_ideb_escola_wide.columns = [
    f"IDEB_{etapa}_{ano}" for etapa, ano in df_ideb_escola_wide.columns
]

df_ideb_escola_wide = df_ideb_escola_wide.reset_index()


# -------------------------------
# 6. Ler base PNME/RDD e pegar controles de 2017
# -------------------------------

df_base = pd.read_csv(
    path_base,
    dtype={"CO_ENTIDADE": str},
    low_memory=False
)

df_base.columns = df_base.columns.str.strip()

df_base["CO_ENTIDADE"] = (
    df_base["CO_ENTIDADE"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.strip()
    .str.zfill(8)
)

df_base["ANO"] = pd.to_numeric(df_base["ANO"], errors="coerce").astype(int)

df_2017 = df_base[df_base["ANO"] == 2017].copy()

cols_renomear = {
    "QT_MAT_FUND": "QT_MAT_FUND_2017",
    "QT_BENEFICIARIOS": "QT_BENEFICIARIOS_2017",
    "pct_bolsa_familia": "pct_bolsa_familia_2017",
    "corte_50": "corte_50_2017",
    "tratamento_real": "tratamento_2017"
}

df_2017 = df_2017.rename(columns=cols_renomear)

cols_controles = [
    "CO_ENTIDADE",
    "tratamento_2017",
    "QT_MAT_FUND_2017",
    "QT_BENEFICIARIOS_2017",
    "pct_bolsa_familia_2017",
    "corte_50_2017",
    "urbana",
    "dep_estadual",
    "dep_municipal"
]

cols_controles = [c for c in cols_controles if c in df_2017.columns]

df_2017_controles = df_2017[cols_controles].copy()

for col in cols_controles:
    if col != "CO_ENTIDADE":
        df_2017_controles[col] = pd.to_numeric(df_2017_controles[col], errors="coerce")

if "QT_MAT_FUND_2017" in df_2017_controles.columns:
    df_2017_controles["ln_QT_MAT_FUND_2017"] = np.where(
        df_2017_controles["QT_MAT_FUND_2017"] > 0,
        np.log(df_2017_controles["QT_MAT_FUND_2017"]),
        np.nan
    )

print("\n==============================")
print("BASE PNME/CONTROLES 2017")
print("==============================")
print(df_2017_controles.shape)

print("\nTratamento 2017:")
print(df_2017_controles["tratamento_2017"].value_counts(dropna=False))

print("\nDuplicatas CO_ENTIDADE nos controles 2017:")
print(df_2017_controles.duplicated("CO_ENTIDADE").sum())


# -------------------------------
# 7. Merge painel IDEB + tratamento/controles 2017
# -------------------------------

df_painel = df_ideb_escola_ano.merge(
    df_2017_controles,
    on="CO_ENTIDADE",
    how="inner"
)

# Variáveis temporais
df_painel["pre_pnme"] = (df_painel["ANO"] <= 2015).astype(int)
df_painel["ano_2017"] = (df_painel["ANO"] == 2017).astype(int)
df_painel["ano_2019"] = (df_painel["ANO"] == 2019).astype(int)

# DiD simples 2015-2019
df_painel["pos_2019"] = (df_painel["ANO"] == 2019).astype(int)

# Interações
df_painel["tratado_x_2017"] = df_painel["tratamento_2017"] * df_painel["ano_2017"]
df_painel["tratado_x_2019"] = df_painel["tratamento_2017"] * df_painel["ano_2019"]
df_painel["tratado_x_pos2019"] = df_painel["tratamento_2017"] * df_painel["pos_2019"]

print("\n==============================")
print("PAINEL IDEB + PNME 2005-2019")
print("==============================")
print(df_painel.shape)
print(df_painel.head())

print("\nAnos no painel:")
print(df_painel["ANO"].value_counts().sort_index())

print("\nEscolas únicas no painel:")
print(df_painel["CO_ENTIDADE"].nunique())

print("\nQuantidade com IDEB por ano:")
print(
    df_painel
    .groupby("ANO")[["IDEB_AI", "IDEB_AF"]]
    .apply(lambda x: x.notna().sum())
)

print("\nTratamento médio por ano entre observações com IDEB_AI:")
print(
    df_painel[df_painel["IDEB_AI"].notna()]
    .groupby("ANO")["tratamento_2017"]
    .mean()
)

print("\nTratamento médio por ano entre observações com IDEB_AF:")
print(
    df_painel[df_painel["IDEB_AF"].notna()]
    .groupby("ANO")["tratamento_2017"]
    .mean()
)


# -------------------------------
# 8. Salvar arquivos
# -------------------------------

df_ideb_long.to_csv(output_ideb_long, index=False, encoding="utf-8-sig")
df_ideb_escola_ano.to_csv(output_ideb_escola_ano, index=False, encoding="utf-8-sig")
df_ideb_escola_wide.to_csv(output_ideb_escola_wide, index=False, encoding="utf-8-sig")
df_painel.to_csv(output_painel, index=False, encoding="utf-8-sig")

print("\nArquivos salvos:")
print(output_ideb_long)
print(output_ideb_escola_ano)
print(output_ideb_escola_wide)
print(output_painel)
# ======================================================
# matching_ideb_controle.py
# Matching de controles por trajetória prévia do IDEB
# Separado para AI e AF
# ======================================================

import pandas as pd
import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Falta instalar scikit-learn. Rode no PowerShell:\n"
        "c:/Users/suporte1/Desktop/lai/.venv/Scripts/python.exe -m pip install scikit-learn"
    )

# -------------------------------
# 1. Caminhos
# -------------------------------

path_painel = r"C:\Users\suporte1\Desktop\lai\painel_ideb_pnme_2005_2019.csv"

output_ai_escolas = r"C:\Users\suporte1\Desktop\lai\matched_escolas_IDEB_AI.csv"
output_af_escolas = r"C:\Users\suporte1\Desktop\lai\matched_escolas_IDEB_AF.csv"

output_ai_painel = r"C:\Users\suporte1\Desktop\lai\matched_painel_IDEB_AI.csv"
output_af_painel = r"C:\Users\suporte1\Desktop\lai\matched_painel_IDEB_AF.csv"

output_balanceamento = r"C:\Users\suporte1\Desktop\lai\balanceamento_matching_IDEB.csv"

# Anos usados para matching
pre_years = [2009, 2011, 2013, 2015]

# Anos mantidos no painel final
anos_painel = [2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019]

# Número de controles por tratada
n_vizinhos = 1

# Com reposição = True: a mesma escola controle pode ser usada para mais de uma tratada.
# Isso preserva mais tratadas e geralmente melhora proximidade.
com_reposicao = True


# -------------------------------
# 2. Funções
# -------------------------------

def preparar_base_wide(df, outcome, etapa):
    """
    Cria base wide por escola:
    CO_ENTIDADE | tratamento_2017 | IDEB_2009 | IDEB_2011 | IDEB_2013 | IDEB_2015 | IDEB_2019
    """
    d = df.copy()
    d = d[df["ANO"].isin(anos_painel)].copy()
    d = d.dropna(subset=[outcome, "tratamento_2017"]).copy()

    wide = (
        d.pivot_table(
            index="CO_ENTIDADE",
            columns="ANO",
            values=outcome,
            aggfunc="first"
        )
        .reset_index()
    )

    wide = wide.rename(columns={ano: f"IDEB_{ano}" for ano in anos_painel})

    controles = [
        "CO_ENTIDADE",
        "tratamento_2017",
        "pct_bolsa_familia_2017",
        "ln_QT_MAT_FUND_2017",
        "urbana",
        "dep_estadual",
        "dep_municipal"
    ]

    controles = [c for c in controles if c in df.columns]

    base_controles = (
        df[controles]
        .drop_duplicates("CO_ENTIDADE")
        .copy()
    )

    wide = wide.merge(base_controles, on="CO_ENTIDADE", how="left")

    # Exigir IDEB nos anos pré usados no matching e no outcome 2019
    cols_obrigatorias = [f"IDEB_{ano}" for ano in pre_years] + ["IDEB_2019", "tratamento_2017"]
    wide = wide.dropna(subset=cols_obrigatorias).copy()

    # Deltas auxiliares para diagnóstico
    wide["delta_2009_2015"] = wide["IDEB_2015"] - wide["IDEB_2009"]
    wide["delta_2011_2015"] = wide["IDEB_2015"] - wide["IDEB_2011"]
    wide["delta_2015_2019"] = wide["IDEB_2019"] - wide["IDEB_2015"]

    wide["etapa"] = etapa
    wide["outcome"] = outcome

    return wide


def calcular_smd(df, vars_balance, grupo_col="tratamento_2017"):
    """
    Calcula diferença padronizada de médias.
    SMD = diferença de médias / desvio padrão combinado.
    """
    linhas = []

    tratados = df[df[grupo_col] == 1]
    controles = df[df[grupo_col] == 0]

    for var in vars_balance:
        mt = tratados[var].mean()
        mc = controles[var].mean()
        vt = tratados[var].var()
        vc = controles[var].var()

        sd_pool = np.sqrt((vt + vc) / 2)

        if sd_pool == 0 or np.isnan(sd_pool):
            smd = np.nan
        else:
            smd = (mt - mc) / sd_pool

        linhas.append({
            "variavel": var,
            "media_tratadas": mt,
            "media_controles": mc,
            "dif_abs": mt - mc,
            "smd": smd
        })

    return pd.DataFrame(linhas)


def fazer_matching_ideb(wide, etapa):
    """
    Matching 1:k por nearest neighbor usando IDEB prévio padronizado.
    """
    match_vars = [f"IDEB_{ano}" for ano in pre_years]

    tratadas = wide[wide["tratamento_2017"] == 1].copy()
    controles = wide[wide["tratamento_2017"] == 0].copy()

    print("\n==============================")
    print(f"MATCHING POR IDEB - {etapa}")
    print("==============================")
    print("Base wide:", wide.shape)
    print("Tratadas elegíveis:", tratadas.shape[0])
    print("Controles elegíveis:", controles.shape[0])
    print("Variáveis de matching:", match_vars)

    scaler = StandardScaler()

    X_controles = scaler.fit_transform(controles[match_vars])
    X_tratadas = scaler.transform(tratadas[match_vars])

    nn = NearestNeighbors(
        n_neighbors=n_vizinhos,
        metric="euclidean"
    )

    nn.fit(X_controles)

    distancias, indices = nn.kneighbors(X_tratadas)

    pares = []

    for i, co_tratada in enumerate(tratadas["CO_ENTIDADE"].values):
        for j in range(n_vizinhos):
            idx_controle = indices[i, j]
            dist = distancias[i, j]

            pares.append({
                "etapa": etapa,
                "CO_ENTIDADE_tratada": co_tratada,
                "CO_ENTIDADE_controle": controles.iloc[idx_controle]["CO_ENTIDADE"],
                "distancia": dist
            })

    df_pares = pd.DataFrame(pares)

    if not com_reposicao:
        # Mantém o melhor par para cada controle e evita reutilização.
        # Observação: isso pode perder tratadas.
        df_pares = (
            df_pares
            .sort_values("distancia")
            .drop_duplicates("CO_ENTIDADE_controle", keep="first")
            .drop_duplicates("CO_ENTIDADE_tratada", keep="first")
            .copy()
        )

    ids_tratadas = df_pares["CO_ENTIDADE_tratada"].unique()
    ids_controles = df_pares["CO_ENTIDADE_controle"].unique()

    matched_tratadas = wide[wide["CO_ENTIDADE"].isin(ids_tratadas)].copy()
    matched_controles = wide[wide["CO_ENTIDADE"].isin(ids_controles)].copy()

    matched_tratadas["grupo_matching"] = "tratada"
    matched_controles["grupo_matching"] = "controle_matched"

    matched = pd.concat([matched_tratadas, matched_controles], ignore_index=True)

    print("\nPares gerados:", df_pares.shape[0])
    print("Tratadas mantidas:", matched_tratadas.shape[0])
    print("Controles únicos matched:", matched_controles.shape[0])
    print("Distância média:", df_pares["distancia"].mean())
    print("Distância mediana:", df_pares["distancia"].median())
    print("Distância p95:", df_pares["distancia"].quantile(0.95))

    vars_balance = match_vars + ["delta_2009_2015", "delta_2011_2015"]

    bal_antes = calcular_smd(wide, vars_balance)
    bal_antes["momento"] = "antes_matching"
    bal_antes["etapa"] = etapa

    bal_depois = calcular_smd(matched, vars_balance)
    bal_depois["momento"] = "depois_matching"
    bal_depois["etapa"] = etapa

    balanceamento = pd.concat([bal_antes, bal_depois], ignore_index=True)

    print("\nBalanceamento antes/depois:")
    print(balanceamento[[
        "etapa", "momento", "variavel",
        "media_tratadas", "media_controles",
        "dif_abs", "smd"
    ]].to_string(index=False))

    return matched, df_pares, balanceamento


def montar_painel_matched(df_painel, matched_escolas, outcome, etapa):
    """
    Mantém no painel original apenas escolas selecionadas no matching.
    """
    ids = matched_escolas["CO_ENTIDADE"].unique()

    painel_matched = df_painel[
        (df_painel["CO_ENTIDADE"].isin(ids)) &
        (df_painel["ANO"].isin(anos_painel)) &
        (df_painel[outcome].notna())
    ].copy()

    painel_matched["etapa_modelo"] = etapa
    painel_matched["outcome_modelo"] = outcome

    return painel_matched


# -------------------------------
# 3. Ler painel
# -------------------------------

df = pd.read_csv(
    path_painel,
    dtype={"CO_ENTIDADE": str},
    low_memory=False
)

df["CO_ENTIDADE"] = df["CO_ENTIDADE"].astype(str).str.zfill(8)
df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").astype(int)
df["tratamento_2017"] = pd.to_numeric(df["tratamento_2017"], errors="coerce")

print("\n==============================")
print("PAINEL CARREGADO")
print("==============================")
print(df.shape)
print(df["ANO"].value_counts().sort_index())


# -------------------------------
# 4. Rodar AI e AF
# -------------------------------

todos_balanceamentos = []

configs = [
    ("AI", "IDEB_AI", output_ai_escolas, output_ai_painel),
    ("AF", "IDEB_AF", output_af_escolas, output_af_painel)
]

for etapa, outcome, output_escolas, output_painel in configs:
    wide = preparar_base_wide(df, outcome, etapa)

    matched, pares, balanceamento = fazer_matching_ideb(wide, etapa)

    painel_matched = montar_painel_matched(df, matched, outcome, etapa)

    matched.to_csv(output_escolas, index=False, encoding="utf-8-sig")
    painel_matched.to_csv(output_painel, index=False, encoding="utf-8-sig")

    pares_path = output_escolas.replace(".csv", "_pares.csv")
    pares.to_csv(pares_path, index=False, encoding="utf-8-sig")

    todos_balanceamentos.append(balanceamento)

    print("\nArquivos salvos para", etapa)
    print(output_escolas)
    print(pares_path)
    print(output_painel)

    print("\nPainel matched:")
    print(painel_matched.shape)
    print("Escolas únicas:", painel_matched["CO_ENTIDADE"].nunique())
    print("Tratamento médio:", painel_matched["tratamento_2017"].mean())
    print("Anos:")
    print(painel_matched["ANO"].value_counts().sort_index())


# -------------------------------
# 5. Salvar balanceamento
# -------------------------------

df_balanceamento = pd.concat(todos_balanceamentos, ignore_index=True)
df_balanceamento.to_csv(output_balanceamento, index=False, encoding="utf-8-sig")

print("\n==============================")
print("BALANCEAMENTO SALVO")
print("==============================")
print(output_balanceamento)

print("\nResumo final balanceamento:")
print(df_balanceamento.to_string(index=False))
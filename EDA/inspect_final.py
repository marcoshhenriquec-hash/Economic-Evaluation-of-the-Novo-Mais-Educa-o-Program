from pathlib import Path
import pandas as pd
import numpy as np

# ======================================================
# Estatísticas descritivas da amostra pareada
# A partir da pasta dos resultados finais
# ======================================================

# Caminho que você passou
RESULTADOS_PATH = Path(
    r"C:\Users\suporte1\Desktop\lai\script_pnme\outputs\10.resultados_finais\resultados_matching_DiD_eventstudy_caliper025.csv"
)

OUT_DIR = RESULTADOS_PATH.parent

#10.resultados_finais\resultados_matching_DiD_eventstudy_caliper025.csv"


OUT_DIR = RESULTADOS_PATH.parent

# Arquivos necessários para montar as descritivas
ARQUIVOS_NECESSARIOS = {
    "AI": "matched_painel_IDEB_AI_caliper025.csv",
    "AF": "matched_painel_IDEB_AF_caliper025.csv",
}


def procurar_arquivo(nome_arquivo: str, base_path: Path) -> Path:
    """
    Procura o arquivo na pasta dos resultados, nas pastas acima
    e na pasta geral C:\\Users\\suporte1\\Desktop\\lai.
    """
    candidatos = [
        base_path.parent,
        base_path.parent.parent,
        base_path.parent.parent.parent,
        Path(r"C:\Users\suporte1\Desktop\lai"),
    ]

    candidatos = list(dict.fromkeys([p for p in candidatos if p.exists()]))

    for pasta in candidatos:
        direto = pasta / nome_arquivo
        if direto.exists():
            return direto

        encontrados = list(pasta.rglob(nome_arquivo))
        if encontrados:
            return encontrados[0]

    raise FileNotFoundError(
        f"Não encontrei {nome_arquivo}. "
        f"Coloque esse arquivo na pasta {base_path.parent} ou ajuste o caminho no script."
    )


def preparar_base_escola(df: pd.DataFrame, etapa: str) -> pd.DataFrame:
    """
    Constrói uma base no nível da escola, usando:
    - características de 2017;
    - IDEB 2015;
    - tratamento PNME 2017.
    """

    outcome = f"IDEB_{etapa}"

    df = df.copy()

    df["CO_ENTIDADE"] = df["CO_ENTIDADE"].astype(str).str.zfill(8)
    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").astype("Int64")
    df["tratamento_2017"] = pd.to_numeric(df["tratamento_2017"], errors="coerce")

    # Controles/variáveis de caracterização
    cols_base = [
        "CO_ENTIDADE",
        "tratamento_2017",
        "pct_bolsa_familia_2017",
        "QT_MAT_FUND_2017",
        "ln_QT_MAT_FUND_2017",
        "urbana",
        "dep_estadual",
        "dep_municipal",
    ]

    cols_base = [c for c in cols_base if c in df.columns]

    base = (
        df[cols_base]
        .drop_duplicates("CO_ENTIDADE")
        .copy()
    )

    # IDEB 2015 da etapa
    ideb_2015 = (
        df[df["ANO"] == 2015][["CO_ENTIDADE", outcome]]
        .drop_duplicates("CO_ENTIDADE")
        .rename(columns={outcome: f"{outcome}_2015"})
    )

    base = base.merge(ideb_2015, on="CO_ENTIDADE", how="left")

    base["etapa"] = etapa

    # Garantir numérico
    for col in base.columns:
        if col not in ["CO_ENTIDADE", "etapa"]:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    return base


def tabela_geral(base: pd.DataFrame, etapa: str) -> pd.DataFrame:
    """
    Estatísticas gerais da amostra pareada.
    """
    outcome_2015 = f"IDEB_{etapa}_2015"

    variaveis = [
        "tratamento_2017",
        "pct_bolsa_familia_2017",
        "QT_MAT_FUND_2017",
        "ln_QT_MAT_FUND_2017",
        "urbana",
        "dep_estadual",
        "dep_municipal",
        outcome_2015,
    ]

    nomes = {
        "tratamento_2017": "Participação no PNME em 2017",
        "pct_bolsa_familia_2017": "Proporção Bolsa Família 2017",
        "QT_MAT_FUND_2017": "Matrículas no fundamental 2017",
        "ln_QT_MAT_FUND_2017": "ln(Matrículas no fundamental 2017)",
        "urbana": "Escola urbana",
        "dep_estadual": "Rede estadual",
        "dep_municipal": "Rede municipal",
        outcome_2015: f"IDEB {etapa} 2015",
    }

    linhas = []

    for var in variaveis:
        if var not in base.columns:
            continue

        x = base[var].dropna()

        linhas.append({
            "etapa": etapa,
            "variavel": nomes.get(var, var),
            "media": x.mean(),
            "desvio_padrao": x.std(),
            "min": x.min(),
            "max": x.max(),
            "n": x.shape[0],
        })

    return pd.DataFrame(linhas)


def tabela_tratadas_controles(base: pd.DataFrame, etapa: str) -> pd.DataFrame:
    """
    Compara médias entre tratadas e controles na amostra pareada.
    """
    outcome_2015 = f"IDEB_{etapa}_2015"

    variaveis = [
        "pct_bolsa_familia_2017",
        "QT_MAT_FUND_2017",
        "ln_QT_MAT_FUND_2017",
        "urbana",
        "dep_estadual",
        "dep_municipal",
        outcome_2015,
    ]

    nomes = {
        "pct_bolsa_familia_2017": "Proporção Bolsa Família 2017",
        "QT_MAT_FUND_2017": "Matrículas no fundamental 2017",
        "ln_QT_MAT_FUND_2017": "ln(Matrículas no fundamental 2017)",
        "urbana": "Escola urbana",
        "dep_estadual": "Rede estadual",
        "dep_municipal": "Rede municipal",
        outcome_2015: f"IDEB {etapa} 2015",
    }

    linhas = []

    controles = base[base["tratamento_2017"] == 0]
    tratadas = base[base["tratamento_2017"] == 1]

    for var in variaveis:
        if var not in base.columns:
            continue

        media_controle = controles[var].mean()
        media_tratada = tratadas[var].mean()

        linhas.append({
            "etapa": etapa,
            "variavel": nomes.get(var, var),
            "controle_media": media_controle,
            "tratadas_media": media_tratada,
            "diferenca_T_C": media_tratada - media_controle,
            "controle_n": controles[var].notna().sum(),
            "tratadas_n": tratadas[var].notna().sum(),
        })

    return pd.DataFrame(linhas)


def formatar_csv(df: pd.DataFrame, path: Path):
    """
    Salva CSV com vírgula decimal amigável ao Excel em PT-BR.
    """
    df.to_csv(path, index=False, encoding="utf-8-sig", sep=";", decimal=",")


def main():
    print("\n==============================")
    print("INICIANDO DESCRITIVAS")
    print("==============================")

    # Só para confirmar que o arquivo-base existe
    if not RESULTADOS_PATH.exists():
        raise FileNotFoundError(f"Não encontrei o arquivo de resultados: {RESULTADOS_PATH}")

    bases_escola = []
    tabelas_gerais = []
    tabelas_tc = []

    for etapa, nome_arquivo in ARQUIVOS_NECESSARIOS.items():
        print(f"\nProcurando arquivo da etapa {etapa}: {nome_arquivo}")
        path = procurar_arquivo(nome_arquivo, RESULTADOS_PATH)
        print("Arquivo encontrado:", path)

        df = pd.read_csv(path, dtype={"CO_ENTIDADE": str}, low_memory=False)

        base = preparar_base_escola(df, etapa)
        bases_escola.append(base)

        tabelas_gerais.append(tabela_geral(base, etapa))
        tabelas_tc.append(tabela_tratadas_controles(base, etapa))

        print(f"Etapa {etapa}")
        print("Escolas únicas:", base["CO_ENTIDADE"].nunique())
        print("Tratadas:", (base["tratamento_2017"] == 1).sum())
        print("Controles:", (base["tratamento_2017"] == 0).sum())

    base_escolas = pd.concat(bases_escola, ignore_index=True)
    desc_geral = pd.concat(tabelas_gerais, ignore_index=True)
    desc_tc = pd.concat(tabelas_tc, ignore_index=True)

    # Arredondar para facilitar leitura
    desc_geral_round = desc_geral.copy()
    desc_tc_round = desc_tc.copy()

    for df in [desc_geral_round, desc_tc_round]:
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].round(4)

    # Outputs
    output_base = OUT_DIR / "base_escolas_descritivas_amostra_pareada.csv"
    output_geral = OUT_DIR / "descritivas_gerais_amostra_pareada.csv"
    output_tc = OUT_DIR / "descritivas_tratadas_controles_amostra_pareada.csv"
    output_xlsx = OUT_DIR / "descritivas_amostra_pareada.xlsx"

    formatar_csv(base_escolas, output_base)
    formatar_csv(desc_geral_round, output_geral)
    formatar_csv(desc_tc_round, output_tc)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        desc_geral_round.to_excel(writer, sheet_name="Descritivas gerais", index=False)
        desc_tc_round.to_excel(writer, sheet_name="Tratadas vs controles", index=False)
        base_escolas.to_excel(writer, sheet_name="Base escola", index=False)

    print("\n==============================")
    print("ARQUIVOS GERADOS")
    print("==============================")
    print(output_base)
    print(output_geral)
    print(output_tc)
    print(output_xlsx)

    print("\n==============================")
    print("PRÉVIA - DESCRITIVAS GERAIS")
    print("==============================")
    print(desc_geral_round.to_string(index=False))

    print("\n==============================")
    print("PRÉVIA - TRATADAS VS CONTROLES")
    print("==============================")
    print(desc_tc_round.to_string(index=False))


if __name__ == "__main__":
    main()

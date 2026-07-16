# ======================================================
# 02_inspect_bf_series.py
# Inspeção da nova base Bolsa Família por série
# ======================================================

import pandas as pd
from pathlib import Path

# -------------------------------
# 1. Definir caminho
# -------------------------------

path_bf = Path(
    r'C:\Users\Usuario\Desktop\ufsc\monografia\data\raw_data\updated_data\2017_a_2019_pbf (1)'
)

files = sorted(path_bf.glob('*'))

print("Arquivos encontrados:")
for f in files:
    print("-", f.name)

# -------------------------------
# 2. Ler arquivos (amostra)
# -------------------------------

for file in files:
    print("\n" + "="*50)
    print(f"LENDO: {file.name}")
    print("="*50)

    try:
        df = pd.read_csv(
            file,
            sep=';',
            encoding='latin1',
            nrows=5
        )

        print("\nColunas:")
        for c in df.columns:
            print("-", repr(c))

        print("\nPreview:")
        print(df.head(3))

    except Exception as e:
        print(f"Erro ao ler {file.name}: {e}")

# -------------------------------
# 3. Ler completo (1 arquivo teste)
# -------------------------------

print("\n" + "="*50)
print("TESTE COMPLETO (1 ARQUIVO)")
print("="*50)

file_test = files[0]

df_full = pd.read_csv(
    file_test,
    sep=';',
    encoding='latin1',
    dtype={'INEP_ESCOLA': str}
)

print("\nShape:", df_full.shape)

print("\nColuna SERIE (valores únicos):")
print(df_full['SERIE'].unique())

print("\nChecando meses:")
cols_meses = [
    'JANAIRO','FEVEREIRO','MARCO','ABRIL','MAIO','JUNHO',
    'JULHO','AGOSTO','SETEMBRO','OUTUBRO','NOVEMBRO','DEZEMBRO'
]

print(df_full[cols_meses].describe())

print("\nDuplicatas (escola-ano-serie):")
print(df_full.duplicated(['INEP_ESCOLA','SERIE','ANO']).sum())

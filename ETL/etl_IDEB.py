# ======================================================
# 07_excel_para_csv_ideb.py
# Converter IDEB Excel para CSV
# ======================================================

import pandas as pd

# -------------------------------
# 1. Caminhos
# -------------------------------

path_ai = r"C:\Users\suporte1\Desktop\lai\IDEB\anos iniciais\divulgacao_anos_iniciais_escolas_2023\divulgacao_anos_iniciais_escolas_2023.csv"

path_af = r"C:\Users\suporte1\Desktop\lai\IDEB\anos finais\divulgacao_anos_finais_escolas_2023.csv"

output_ai = r"C:\Users\suporte1\Desktop\lai\IDEB\ideb_anos_iniciais.csv"

output_af = r"C:\Users\suporte1\Desktop\lai\IDEB\ideb_anos_finais.csv"

# -------------------------------
# 2. Ler Excel
# -------------------------------

df_ai = pd.read_csv(path_ai, sep=";", encoding="utf-8-sig", dtype=str)

df_af = pd.read_csv(path_af, sep=";", encoding="utf-8-sig", dtype=str)

# -------------------------------
# 3. Salvar CSV
# -------------------------------
# 4. Prints
# -------------------------------

print("\n==============================")
print("IDEB ANOS INICIAIS")
print("==============================")

print("\nShape:")
print(df_ai.shape)

print("\nHead:")
print(df_ai.head())

print("\nColunas:")
print(df_ai.columns.tolist())

print("\nCSV salvo em:")
print(output_ai)


print("\n==============================")
print("IDEB ANOS FINAIS")
print("==============================")

print("\nShape:")
print(df_af.shape)

print("\nHead:")
print(df_af.head())

print("\nColunas:")
print(df_af.columns.tolist())

print("\nCSV salvo em:")
print(output_af)

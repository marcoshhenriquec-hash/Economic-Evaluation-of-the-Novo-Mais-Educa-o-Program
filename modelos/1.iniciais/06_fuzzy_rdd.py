#FUZZY RDD (2SLS)

# ======================================================
# 03_rdd_fuzzy.py
# Fuzzy RDD via 2SLS
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.iv import IV2SLS

# -------------------------------
# 1. Carregar base final
# -------------------------------

df = pd.read_csv(
    r'C:\Users\Usuario\Desktop\ufsc\monografia\data\processed_data\base_rdd_escolas_2017_2019.csv'
)

# -------------------------------
# 2. Running variable centrada
# -------------------------------

df['running_c'] = df['pct_bolsa_familia'] - 0.5

# -------------------------------
# 3. Restrição local (janela)
# -------------------------------

df_local = df[
    (df['pct_bolsa_familia'] >= 0.4) &
    (df['pct_bolsa_familia'] <= 0.6)
].copy()

print("Observações na janela:", len(df_local))

# -------------------------------
# 4. DEFINIR VARIÁVEIS
# -------------+-+-+------------------

# 🔴 VOCÊ PRECISA AJUSTAR ISSO:
Y = 'QT_MAT_FUND'   # outcome (trocar depois)
D = 'tratamento_real'  # variável de tratamento (ex: participação PNME)
Z = 'corte_50'      # instrumento

# -------------------------------
# 5. Modelo fuzzy RDD (2SLS)
# -------------------------------

modelo = IV2SLS.from_formula(
    f"{Y} ~ 1 + running_c + running_c:corte_50 + [{D} ~ {Z}]",
    data=df_local
).fit(cov_type='robust')

print(modelo.summary)

# -------------------------------
# 6. First stage (força do instrumento)
# -------------------------------

print("\nFirst stage:")
print(modelo.first_stage)

# -------------------------------
# 7. Visual simples (RDD)
# -------------------------------

plt.figure(figsize=(8,5))
plt.scatter(df_local['pct_bolsa_familia'], df_local[Y], alpha=0.2)
plt.axvline(0.5)
plt.title("RDD Fuzzy - visual")
plt.show()
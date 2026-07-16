import pandas as pd

censo_path = r"C:\Users\suporte1\Desktop\lai\censos escolares\censo_2017-20260605T132546Z-3-001\censo_2017\microdados_censo_escolar_2017\microdados_ed_basica_2017\dados\md5_microdados_ed_basica_2017.txt"

df_censo = pd.read_csv(censo_path)

print("\nColunas censo:")
print(df_censo.columns.tolist())

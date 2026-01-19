import pandas as pd


path_censo = r"C:\Users\Usuario\Desktop\ufsc\monografia\data\raw_data"


#colunas relevantes p/ não ler o df inteira
cols_censo = [ 'NU_ANO_CENSO',
    'CO_ENTIDADE',
    'QT_MAT_FUND',
    'TP_LOCALIZACAO',
    'TP_DEPENDENCIA']

anos = range(2015,2020)

#lista para juntar os dfs de todos o anos
dfs = []

for ano in anos:
    print(f"Lendo Censo {ano}...")
    
    path =  rf"{path_censo}\censo_{ano}\microdados_censo_escolar_{ano}\microdados_ed_basica_{ano}\dados\microdados_ed_basica_{ano}.csv"
    
    df_ano = pd.read_csv(path,sep=';',
                         encoding='latin1',
                         usecols=cols_censo,
                         low_memory = False)
    
#filtrar escolas com ensino fundamental 
    df_ano = df_ano[df_ano['QT_MAT_FUND']>0].copy()

#transformar variáveis em dummies
    df_ano['urbana'] = (df_ano['TP_LOCALIZACAO']==1).astype(int)
    df_ano['dep_estadual'] = (df_ano['TP_DEPENDENCIA'] == 2).astype(int)
    df_ano['dep_municipal'] = (df_ano['TP_DEPENDENCIA'] == 3).astype(int)

#excluir rede privada(não contemplada pelo PNME)
    df_ano = df_ano[df_ano['TP_DEPENDENCIA']!=4].copy()

    df_ano = df_ano[[ 'NU_ANO_CENSO',
                    'CO_ENTIDADE',
                    'QT_MAT_FUND',
                    'urbana',
                    'dep_estadual',
                    'dep_municipal']].copy()
    
    dfs.append(df_ano)

    del df_ano

#concatenar dfs

df_15_19 = pd.concat(dfs,ignore_index=True)

print("todos os anos concatenados")

df_15_19.to_csv(r"C:\Users\Usuario\Desktop\ufsc\monografia\data\data_processed\censo_escolar_2015_2019.csv")

print("arquivo salvo")
  
    
    
    




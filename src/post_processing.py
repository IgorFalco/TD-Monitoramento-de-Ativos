import pandas as pd

import pandas as pd
from pathlib import Path

def read_csv(path_folder: str, pattern_file: str) -> pd.DataFrame:
    pasta = Path(path_folder)
    
    files = list(pasta.glob(pattern_file))
    
    if not files:
        print(f"Nenhum arquivo encontrado com o padrão '{pattern_file}' em '{path_folder}'.")
        return pd.DataFrame()

    print(f"Arquivos encontrados ({len(files)}):")
    for f in files:
        print(f" - {f.name}")
        
    dataframes = []
    
    for file in files:
        try:
            df = pd.read_csv(file, sep=';', decimal=',')
            dataframes.append(df)
        except Exception as e:
            print(f"Erro ao ler o arquivo {file.name}: {e}")

    if not dataframes:
        print("Nenhum arquivo pôde ser lido com sucesso.")
        return pd.DataFrame()

    df_merge = pd.concat(dataframes, ignore_index=True)
    
    return df_merge


if __name__ == "__main__":
    path_folder = "result" 
    pattern_file = "fronteira_epsilon_*.csv" 

    final_df = read_csv(path_folder, pattern_file)

    statics = final_df.groupby('f2_equipes')['f1_distancia'].agg(['min', 'mean', 'std'])
    print(statics)


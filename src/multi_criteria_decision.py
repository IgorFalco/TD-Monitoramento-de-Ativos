import numpy as np
import pandas as pd


class AHPAnalysis:
    def __init__(self, df_datas, type_objectives, matrix_comp):
        self.df_raw = df_datas.copy()
        self.df_normalized = pd.DataFrame(index=df_datas.index)
        self.weight_criteria = None
        self.cr = None # Razão de Consistência
        self.final_ranking = None
        self.type_objectives = type_objectives
        self.calculate_matrix_weight(matrix_comp)
        self.consistency_calculate(matrix_comp)

    def consistency_calculate(self, matrix):
        # 3. Cálculo da Consistência (CR)
        matrix = np.array(matrix)
        len_matrix = matrix.shape[0]
        vetor_soma = np.dot(matrix, self.weight_criteria)
        lambda_max = (vetor_soma / self.weight_criteria).mean()
        ci = (lambda_max - len_matrix) / (len_matrix - 1)
        
        # Tabela RI random index (Saaty)
        dict_ri = {1: 0.00, 2: 0.00, 3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35, 8: 1.40, 9: 1.49}
        ri = dict_ri.get(len_matrix, 1.49)
        
        self.cr = ci / ri if ri != 0 else 0

    def calculate_matrix_weight(self, matriz_saaty):
        """
        Recebe a matriz de comparação par-a-par (Saaty) e calcula os pesos (autovetor).
        Também calcula a Razão de Consistência (CR).
        """
        matrix = np.array(matriz_saaty)
        n = matrix.shape[0]
        
        # Verifica se o tamanho da matriz bate com o número de critérios configurados
        if n != len(self.type_objectives):
            raise ValueError(f"A matriz é {n}x{n}, mas você configurou {len(self.type_objectives)} critérios.")

        # 1. Normalização da Matriz
        col_sums = matrix.sum(axis=0)
        matrix_norm = matrix / col_sums
        
        # 2. Vetor de Pesos (Média das linhas)
        self.weight_criteria = matrix_norm.mean(axis=1)
        
        # Retorna um Series bonitinho para visualização
        return pd.Series(self.weight_criteria, index=self.type_objectives.keys())

    def _column_normalize(self, serie, type):
        """Método interno para normalizar valores (minimização ou maximização)."""
        values = np.array(serie)
        
        if type == 'min':
            # Para minimizar (quanto menor melhor), usamos o inverso (1/x)
            # Adiciona um epsilon minúsculo para evitar divisão por zero
            values = np.where(values == 0, 1e-9, values)
            inv_values = 1 / values
            return inv_values / inv_values.sum()
        
        elif type == 'max':
            # Para maximizar (quanto maior melhor)
            return values / values.sum()
        
        else:
            raise ValueError("Tipo deve ser 'min' ou 'max'")

    def execute_analisys(self):
        """
        Executa a normalização dos dados reais e calcula o ranking final.
        """
        if self.weight_criteria is None:
            raise Exception("Você precisa calcular os pesos da matriz antes de executar a análise.")

        ordered_column = list(self.type_objectives.keys())

        # 1. Normaliza as Alternativas (Dados Quantitativos)
        for col, type in self.type_objectives.items():
            self.df_normalized[col] = self._column_normalize(self.df_raw[col], type)

        # 2. Cálculo do Score Global (Soma Ponderada)
        # Garante que a ordem dos pesos bate com a ordem das colunas
        matriz_norm_values = self.df_normalized[ordered_column].values
        scores = np.dot(matriz_norm_values, self.weight_criteria)
        
        # 3. Montagem do Resultado
        self.final_ranking = self.df_raw.copy()
        self.final_ranking['AHP_Score'] = scores
        self.final_ranking['Posição'] = self.final_ranking['AHP_Score'].rank(ascending=False).astype(int)
        self.final_ranking = self.final_ranking.sort_values('Posição')
        
        return self.final_ranking

if __name__ == "__main__":
    data_input = {
        'distance_total': np.random.randint(150, 600, 20),  # Km
        'num_teams': np.random.randint(2, 8, 20),          # Qtd
        'sla_atendimento': np.round(np.random.uniform(0.5, 4.0, 20), 1), # Horas
        'trajetory_risk': np.random.randint(1, 11, 20)     # Nota 1-10
    }
    type_objectives = {
        'distance_total': 'min',
        'num_teams': 'min',
        'sla_atendimento': 'min',
        'trajetory_risk': 'min'
    }
    # 1, 3, 5, 7, 9
    comp_matrix = [
        # dist, equi, sla, risco
        [1.0, 1/3, 1/7, 1/5], # Distancia (menos importante)
        [3.0, 1.0, 1/5, 1/3], # Equipes (terceiro mais importante)
        [7.0, 5.0, 1.0, 3], # SLA (mais importante)
        [5.0, 3.0, 1/3, 1.0]  # Risco de trajetoria(Segundo mais importante)
    ]

    df_entrada = pd.DataFrame(data_input)
    ahp = AHPAnalysis(df_entrada, type_objectives, comp_matrix)
    

    result = ahp.execute_analisys()
    print(result)

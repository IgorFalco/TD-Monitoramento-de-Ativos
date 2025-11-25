import numpy as np
import pandas as pd


class PROMETHEEAnalysis:
    """
    Implementação do PROMETHEE II para ranking completo de alternativas.
    
    PROMETHEE (Preference Ranking Organization METHod for Enrichment Evaluation)
    é um método de decisão multicritério baseado em comparações par-a-par.
    """
    
    def __init__(self, df_data, criteria_config, weights):
        """
        Inicializa a análise PROMETHEE.
        
        Args:
            df_data: DataFrame com alternativas (linhas) e critérios (colunas)
            criteria_config: dict {criterio: {'type': 'max'|'min', 'preference': func}}
                preference pode ser: 'usual', 'linear', 'level', 'gaussian'
            weights: dict {criterio: peso} ou array com pesos na ordem das colunas
        """
        self.df_raw = df_data.copy()
        self.criteria_config = criteria_config
        
        # Normaliza pesos
        if isinstance(weights, dict):
            self.weights = np.array([weights[col] for col in df_data.columns])
        else:
            self.weights = np.array(weights)
        self.weights = self.weights / self.weights.sum()
        
        self.preference_matrix = None
        self.phi_plus = None  # Fluxo positivo
        self.phi_minus = None  # Fluxo negativo
        self.phi_net = None  # Fluxo líquido
        self.final_ranking = None
    
    def _preference_function(self, diff, criterion, pref_type='usual', q=None, p=None, s=None):
        """
        Calcula grau de preferência P(a,b) para um critério.
        
        Args:
            diff: diferença entre alternativas (a - b)
            criterion: nome do critério
            pref_type: tipo de função ('usual', 'linear', 'level', 'gaussian')
            q: limiar de indiferença
            p: limiar de preferência estrita
            s: parâmetro para gaussian
        """
        if pref_type == 'usual':
            # Preferência tipo degrau (0 ou 1)
            return 1.0 if diff > 0 else 0.0
        
        elif pref_type == 'linear':
            # Preferência linear entre q e p
            if q is None:
                q = 0
            if p is None:
                p = np.std(self.df_raw[criterion])
            
            if diff <= q:
                return 0.0
            elif diff >= p:
                return 1.0
            else:
                return (diff - q) / (p - q)
        
        elif pref_type == 'level':
            # Preferência com patamar
            if q is None:
                q = np.std(self.df_raw[criterion]) * 0.3
            if p is None:
                p = np.std(self.df_raw[criterion])
            
            if diff <= q:
                return 0.0
            elif diff <= p:
                return 0.5
            else:
                return 1.0
        
        elif pref_type == 'gaussian':
            # Preferência gaussiana
            if s is None:
                s = np.std(self.df_raw[criterion]) * 0.5
            
            if diff <= 0:
                return 0.0
            else:
                return 1.0 - np.exp(-(diff**2) / (2 * s**2))
        
        else:
            raise ValueError(f"Tipo de preferência '{pref_type}' não reconhecido")
    
    def _calculate_preference_index(self, alt_a, alt_b):
        """
        Calcula índice de preferência agregado π(a,b) considerando todos os critérios.
        """
        pi = 0.0
        
        for idx, criterion in enumerate(self.df_raw.columns):
            config = self.criteria_config[criterion]
            val_a = alt_a[criterion]
            val_b = alt_b[criterion]
            
            # Calcula diferença respeitando tipo de otimização
            if config['type'] == 'max':
                diff = val_a - val_b  # Quanto maior, melhor
            else:  # 'min'
                diff = val_b - val_a  # Quanto menor, melhor
            
            # Calcula preferência para este critério
            pref_params = config.get('params', {})
            pref = self._preference_function(
                diff, 
                criterion, 
                pref_type=config.get('preference', 'usual'),
                **pref_params
            )
            
            # Pondera pela importância do critério
            pi += self.weights[idx] * pref
        
        return pi
    
    def execute_analysis(self):
        """
        Executa análise PROMETHEE II completa.
        """
        n_alternatives = len(self.df_raw)
        
        # Matriz de preferências agregadas π(a,b)
        self.preference_matrix = np.zeros((n_alternatives, n_alternatives))
        
        for i in range(n_alternatives):
            for j in range(n_alternatives):
                if i != j:
                    self.preference_matrix[i, j] = self._calculate_preference_index(
                        self.df_raw.iloc[i],
                        self.df_raw.iloc[j]
                    )
        
        # Fluxos de sobreclassificação
        self.phi_plus = self.preference_matrix.sum(axis=1) / (n_alternatives - 1)   # Fluxo positivo
        self.phi_minus = self.preference_matrix.sum(axis=0) / (n_alternatives - 1)  # Fluxo negativo
        self.phi_net = self.phi_plus - self.phi_minus  # Fluxo líquido (PROMETHEE II)
        
        # Monta resultado final
        self.final_ranking = self.df_raw.copy()
        self.final_ranking['Phi_Plus'] = self.phi_plus
        self.final_ranking['Phi_Minus'] = self.phi_minus
        self.final_ranking['Phi_Net'] = self.phi_net
        self.final_ranking['Posição'] = self.phi_net.argsort()[::-1].argsort() + 1
        self.final_ranking = self.final_ranking.sort_values('Posição')
        
        return self.final_ranking
    
    def get_incomparable_pairs(self, threshold=0.01):
        """
        Identifica pares de alternativas incomparáveis (fluxos líquidos muito próximos).
        """
        if self.phi_net is None:
            raise Exception("Execute a análise antes de verificar incomparabilidades")
        
        incomparable = []
        n = len(self.phi_net)
        
        for i in range(n):
            for j in range(i+1, n):
                if abs(self.phi_net[i] - self.phi_net[j]) < threshold:
                    incomparable.append((i, j, self.phi_net[i], self.phi_net[j]))
        
        return incomparable


if __name__ == "__main__":
    # Exemplo de uso
    data = {
        'distance': [685, 745, 802, 1117, 1417, 3420],
        'teams': [8, 7, 5, 3, 2, 1],
        'robustness': [0.92, 0.88, 0.85, 0.78, 0.72, 0.65],
        'reliability': [0.95, 0.93, 0.90, 0.85, 0.82, 0.75]
    }
    
    df = pd.DataFrame(data)
    
    # Configuração dos critérios
    criteria_config = {
        'distance': {'type': 'min', 'preference': 'linear', 'params': {'q': 50, 'p': 200}},
        'teams': {'type': 'min', 'preference': 'usual'},
        'robustness': {'type': 'max', 'preference': 'linear', 'params': {'q': 0.02, 'p': 0.1}},
        'reliability': {'type': 'max', 'preference': 'linear', 'params': {'q': 0.02, 'p': 0.1}}
    }
    
    # Pesos (exemplo: SLA mais importante)
    weights = [0.2, 0.3, 0.25, 0.25]
    
    promethee = PROMETHEEAnalysis(df, criteria_config, weights)
    result = promethee.execute_analysis()
    
    print("\n=== PROMETHEE II - Ranking ===")
    print(result[['distance', 'teams', 'Phi_Net', 'Posição']])
    
    print(f"\nMelhor solução: Alternativa {result.iloc[0].name}")

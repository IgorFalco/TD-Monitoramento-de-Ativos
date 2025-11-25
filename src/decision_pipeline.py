"""
Módulo de Decisão Multicritério - Integração AHP e PROMETHEE
Consolida fronteiras Pareto, define critérios adicionais e aplica métodos de decisão.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multi_criteria_decision import AHPAnalysis
from promethee import PROMETHEEAnalysis


class ParetoConsolidator:
    """Consolida fronteiras Pareto de múltiplas execuções."""
    
    def __init__(self, results_dir='result'):
        self.results_dir = results_dir
        self.all_solutions = []
        self.pareto_front = None
        self.selected_solutions = None
    
    def load_all_frontiers(self, method='soma_ponderada'):
        """
        Carrega todos os CSVs de um método e unifica.
        
        Args:
            method: 'soma_ponderada' ou 'epsilon_restrito'
        """
        pattern = os.path.join(self.results_dir, f'fronteira_{method}_*.csv')
        files = glob.glob(pattern)
        
        if not files:
            # Debug: mostra caminho completo e arquivos disponíveis
            abs_pattern = os.path.abspath(pattern)
            available = glob.glob(os.path.join(self.results_dir, '*.csv'))
            raise FileNotFoundError(
                f"Nenhum arquivo encontrado para '{method}'\n"
                f"  Padrão procurado: {abs_pattern}\n"
                f"  Arquivos disponíveis: {[os.path.basename(f) for f in available]}"
            )
        
        print(f"\n📁 Carregando {len(files)} arquivos de {method}...")
        
        dfs = []
        for f in files:
            df = pd.read_csv(f, sep=';', decimal=',')
            dfs.append(df)
        
        # Une todas as fronteiras
        combined = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicatas exatas (mesmo f1 e f2)
        combined = combined.drop_duplicates(subset=['f1_distancia', 'f2_equipes'])
        
        print(f"   ✓ Total de soluções únicas: {len(combined)}")
        self.all_solutions = combined
        return combined
    
    def filter_non_dominated(self):
        """Filtra apenas soluções não-dominadas da união."""
        if self.all_solutions is None or len(self.all_solutions) == 0:
            raise Exception("Carregue as fronteiras antes de filtrar")
        
        df = self.all_solutions.copy()
        non_dominated_idx = []
        
        for i in range(len(df)):
            sol_i = df.iloc[i]
            dominated = False
            
            for j in range(len(df)):
                if i == j:
                    continue
                
                sol_j = df.iloc[j]
                
                # sol_i é dominada por sol_j se:
                # - sol_j é melhor ou igual em todos os objetivos
                # - sol_j é estritamente melhor em pelo menos um
                f1_worse = sol_i['f1_distancia'] >= sol_j['f1_distancia']
                f2_worse = sol_i['f2_equipes'] >= sol_j['f2_equipes']
                at_least_one_better = (sol_i['f1_distancia'] > sol_j['f1_distancia'] or 
                                      sol_i['f2_equipes'] > sol_j['f2_equipes'])
                
                if f1_worse and f2_worse and at_least_one_better:
                    dominated = True
                    break
            
            if not dominated:
                non_dominated_idx.append(i)
        
        self.pareto_front = df.iloc[non_dominated_idx].reset_index(drop=True)
        print(f"\n🎯 Fronteira Pareto global: {len(self.pareto_front)} soluções não-dominadas")
        return self.pareto_front
    
    def select_representative(self, max_solutions=20):
        """
        Seleciona as soluções mais representativas (bem distribuídas) da fronteira.
        Algoritmo: maximiza distância mínima entre pontos selecionados.
        """
        if self.pareto_front is None or len(self.pareto_front) == 0:
            raise Exception("Filtre as soluções não-dominadas primeiro")
        
        if len(self.pareto_front) <= max_solutions:
            self.selected_solutions = self.pareto_front.copy()
            print(f"   ℹ️ Fronteira tem {len(self.pareto_front)} soluções, todas serão usadas")
            return self.selected_solutions
        
        # Ordena por f1
        sorted_df = self.pareto_front.sort_values('f1_distancia').reset_index(drop=True)
        
        # Sempre inclui os extremos
        selected_idx = [0, len(sorted_df) - 1]
        
        # Normaliza para cálculo de distâncias
        f1_norm = (sorted_df['f1_distancia'] - sorted_df['f1_distancia'].min()) / \
                  (sorted_df['f1_distancia'].max() - sorted_df['f1_distancia'].min() + 1e-9)
        f2_norm = (sorted_df['f2_equipes'] - sorted_df['f2_equipes'].min()) / \
                  (sorted_df['f2_equipes'].max() - sorted_df['f2_equipes'].min() + 1e-9)
        
        # Seleciona pontos que maximizam distância mínima aos já selecionados
        while len(selected_idx) < max_solutions:
            max_min_dist = -1
            best_candidate = -1
            
            for i in range(len(sorted_df)):
                if i in selected_idx:
                    continue
                
                # Distância mínima deste candidato aos já selecionados
                min_dist = float('inf')
                for j in selected_idx:
                    dist = np.sqrt((f1_norm[i] - f1_norm[j])**2 + 
                                  (f2_norm[i] - f2_norm[j])**2)
                    min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = i
            
            if best_candidate == -1:
                break
            
            selected_idx.append(best_candidate)
        
        self.selected_solutions = sorted_df.iloc[selected_idx].reset_index(drop=True)
        print(f"   ✓ Selecionadas {len(self.selected_solutions)} soluções representativas")
        return self.selected_solutions


class AdditionalCriteriaGenerator:
    """
    Gera critérios adicionais para as soluções Pareto.
    Critérios devem ser conflitantes com os objetivos originais.
    """
    
    @staticmethod
    def calculate_robustness(df_solutions, seed=42):
        """
        Robustez: quão estável é a solução ante variações nos parâmetros.
        Simulação: perturbações aleatórias nas distâncias e recálculo de f1.
        Menor variação relativa = mais robusta.
        
        Proxy: soluções com menos equipes e distâncias intermediárias tendem a ser mais robustas.
        """
        np.random.seed(seed)
        
        robustness_scores = []
        for idx, row in df_solutions.iterrows():
            f1 = row['f1_distancia']
            f2 = row['f2_equipes']
            
            # Simulação simplificada: variação proporcional à complexidade
            # Mais equipes = mais pontos de falha = menos robusto
            # Distâncias muito altas ou muito baixas = extremos menos robustos
            
            # Normaliza f1 e f2
            f1_norm = (f1 - df_solutions['f1_distancia'].min()) / \
                     (df_solutions['f1_distancia'].max() - df_solutions['f1_distancia'].min() + 1e-9)
            f2_norm = (f2 - df_solutions['f2_equipes'].min()) / \
                     (df_solutions['f2_equipes'].max() - df_solutions['f2_equipes'].min() + 1e-9)
            
            # Robustez penaliza extremos e muitas equipes
            # Valor entre 0 e 1 (maior = mais robusto)
            robustness = 1.0 - 0.4 * f2_norm - 0.3 * abs(0.5 - f1_norm) + np.random.uniform(-0.05, 0.05)
            robustness = np.clip(robustness, 0.5, 0.98)
            robustness_scores.append(robustness)
        
        return np.array(robustness_scores)
    
    @staticmethod
    def calculate_reliability(df_solutions, seed=42):
        """
        Confiabilidade: probabilidade de atender todos os ativos com sucesso.
        Menos equipes = menos redundância = menor confiabilidade.
        Distâncias maiores = maior risco de falha = menor confiabilidade.
        """
        np.random.seed(seed + 10)
        
        reliability_scores = []
        for idx, row in df_solutions.iterrows():
            f1 = row['f1_distancia']
            f2 = row['f2_equipes']
            
            # Normaliza
            f1_norm = (f1 - df_solutions['f1_distancia'].min()) / \
                     (df_solutions['f1_distancia'].max() - df_solutions['f1_distancia'].min() + 1e-9)
            f2_norm = (f2 - df_solutions['f2_equipes'].min()) / \
                     (df_solutions['f2_equipes'].max() - df_solutions['f2_equipes'].min() + 1e-9)
            
            # Confiabilidade cresce com mais equipes e distâncias menores
            reliability = 0.95 - 0.25 * f1_norm + 0.15 * f2_norm + np.random.uniform(-0.03, 0.03)
            reliability = np.clip(reliability, 0.65, 0.98)
            reliability_scores.append(reliability)
        
        return np.array(reliability_scores)
    
    @staticmethod
    def calculate_risk_score(df_solutions, seed=42):
        """
        Risco de falha operacional: pontuação de 1 a 10 (menor = melhor).
        Distâncias maiores e menos equipes aumentam o risco.
        """
        np.random.seed(seed + 20)
        
        risk_scores = []
        for idx, row in df_solutions.iterrows():
            f1 = row['f1_distancia']
            f2 = row['f2_equipes']
            
            # Normaliza
            f1_norm = (f1 - df_solutions['f1_distancia'].min()) / \
                     (df_solutions['f1_distancia'].max() - df_solutions['f1_distancia'].min() + 1e-9)
            f2_norm = (f2 - df_solutions['f2_equipes'].min()) / \
                     (df_solutions['f2_equipes'].max() - df_solutions['f2_equipes'].min() + 1e-9)
            
            # Risco aumenta com distância e diminui com equipes
            risk = 2.0 + 6.0 * f1_norm - 2.0 * f2_norm + np.random.uniform(-0.3, 0.3)
            risk = np.clip(risk, 1.0, 10.0)
            risk_scores.append(risk)
        
        return np.array(risk_scores)
    
    @staticmethod
    def calculate_flexibility(df_solutions, seed=42):
        """
        Flexibilidade: capacidade de adaptação a novos cenários.
        Mais equipes = maior flexibilidade para redistribuir ativos.
        """
        np.random.seed(seed + 30)
        
        flexibility_scores = []
        for idx, row in df_solutions.iterrows():
            f2 = row['f2_equipes']
            
            # Normaliza
            f2_norm = (f2 - df_solutions['f2_equipes'].min()) / \
                     (df_solutions['f2_equipes'].max() - df_solutions['f2_equipes'].min() + 1e-9)
            
            # Flexibilidade proporcional ao número de equipes
            flexibility = 0.6 + 0.35 * f2_norm + np.random.uniform(-0.05, 0.05)
            flexibility = np.clip(flexibility, 0.55, 0.98)
            flexibility_scores.append(flexibility)
        
        return np.array(flexibility_scores)
    
    @classmethod
    def enrich_solutions(cls, df_solutions):
        """Adiciona todos os critérios extras ao DataFrame."""
        df_enriched = df_solutions.copy()
        
        df_enriched['robustez'] = cls.calculate_robustness(df_solutions)
        df_enriched['confiabilidade'] = cls.calculate_reliability(df_solutions)
        df_enriched['risco_operacional'] = cls.calculate_risk_score(df_solutions)
        df_enriched['flexibilidade'] = cls.calculate_flexibility(df_solutions)
        
        return df_enriched


def run_multicriteria_decision(results_dir='result', method='soma_ponderada', max_solutions=20):
    """
    Pipeline completo de decisão multicritério.
    
    Returns:
        dict com resultados de AHP, PROMETHEE e solução escolhida
    """
    print("\n" + "="*80)
    print("🎯 ANÁLISE DE DECISÃO MULTICRITÉRIO")
    print("="*80)
    
    # 1. Consolida fronteiras Pareto
    print("\n--- Etapa 1: Consolidação das Fronteiras Pareto ---")
    consolidator = ParetoConsolidator(results_dir)
    consolidator.load_all_frontiers(method)
    consolidator.filter_non_dominated()
    selected = consolidator.select_representative(max_solutions)
    
    # 2. Gera critérios adicionais
    print("\n--- Etapa 2: Geração de Critérios Adicionais ---")
    enriched = AdditionalCriteriaGenerator.enrich_solutions(selected)
    
    print("\nCritérios de Decisão:")
    print("  1. f1_distancia (km) - MINIMIZAR")
    print("  2. f2_equipes (qtd) - MINIMIZAR")
    print("  3. robustez (0-1) - MAXIMIZAR")
    print("  4. confiabilidade (0-1) - MAXIMIZAR")
    print("  5. risco_operacional (1-10) - MINIMIZAR")
    print("  6. flexibilidade (0-1) - MAXIMIZAR")
    
    # Prepara dados para métodos de decisão
    criteria_columns = ['f1_distancia', 'f2_equipes', 'robustez', 
                       'confiabilidade', 'risco_operacional', 'flexibilidade']
    decision_data = enriched[criteria_columns].copy()
    
    # 3. Aplica AHP
    print("\n--- Etapa 3: Método AHP ---")
    
    # Matriz de comparação par-a-par (Saaty)
    # Ordem: f1, f2, robustez, confiabilidade, risco, flexibilidade
    # Prioridades: confiabilidade > robustez > f2 > risco > flexibilidade > f1
    ahp_matrix = [
        [1.0,   1/3,  1/5,  1/7,  1/3,  1/2],  # f1_distancia
        [3.0,   1.0,  1/3,  1/5,  1/2,  2.0],  # f2_equipes
        [5.0,   3.0,  1.0,  1/3,  3.0,  4.0],  # robustez
        [7.0,   5.0,  3.0,  1.0,  5.0,  6.0],  # confiabilidade (mais importante)
        [3.0,   2.0,  1/3,  1/5,  1.0,  2.0],  # risco_operacional
        [2.0,   1/2,  1/4,  1/6,  1/2,  1.0]   # flexibilidade
    ]
    
    type_objectives = {
        'f1_distancia': 'min',
        'f2_equipes': 'min',
        'robustez': 'max',
        'confiabilidade': 'max',
        'risco_operacional': 'min',
        'flexibilidade': 'max'
    }
    
    ahp = AHPAnalysis(decision_data, type_objectives, ahp_matrix)
    ahp_result = ahp.execute_analisys()
    
    print(f"\nPesos dos Critérios (AHP):")
    for i, col in enumerate(criteria_columns):
        print(f"  {col}: {ahp.weight_criteria[i]:.4f}")
    print(f"\nRazão de Consistência (CR): {ahp.cr:.4f}")
    if ahp.cr < 0.10:
        print("  ✓ Matriz consistente (CR < 0.10)")
    else:
        print("  ⚠️ Matriz inconsistente (CR >= 0.10) - revisar comparações")
    
    # 4. Aplica PROMETHEE II
    print("\n--- Etapa 4: Método PROMETHEE II ---")
    
    # Configuração das funções de preferência
    promethee_config = {
        'f1_distancia': {
            'type': 'min',
            'preference': 'linear',
            'params': {'q': 50, 'p': 300}  # Indiferença até 50km, preferência estrita após 300km
        },
        'f2_equipes': {
            'type': 'min',
            'preference': 'usual'  # Preferência binária (qualquer diferença conta)
        },
        'robustez': {
            'type': 'max',
            'preference': 'linear',
            'params': {'q': 0.02, 'p': 0.10}
        },
        'confiabilidade': {
            'type': 'max',
            'preference': 'linear',
            'params': {'q': 0.02, 'p': 0.10}
        },
        'risco_operacional': {
            'type': 'min',
            'preference': 'linear',
            'params': {'q': 0.5, 'p': 2.0}
        },
        'flexibilidade': {
            'type': 'max',
            'preference': 'linear',
            'params': {'q': 0.03, 'p': 0.12}
        }
    }
    
    # Usa os mesmos pesos do AHP para comparação justa
    promethee_weights = ahp.weight_criteria
    
    promethee = PROMETHEEAnalysis(decision_data, promethee_config, promethee_weights)
    promethee_result = promethee.execute_analysis()
    
    print(f"\nPesos dos Critérios (PROMETHEE - mesmo do AHP):")
    for i, col in enumerate(criteria_columns):
        print(f"  {col}: {promethee_weights[i]:.4f}")
    
    # 5. Comparação dos métodos
    print("\n--- Etapa 5: Comparação AHP vs PROMETHEE ---")
    
    comparison = pd.DataFrame({
        'Solução': range(len(enriched)),
        'f1 (km)': enriched['f1_distancia'].values,
        'f2 (equipes)': enriched['f2_equipes'].values,
        'Rank_AHP': ahp_result['Posição'].values,
        'Score_AHP': ahp_result['AHP_Score'].values,
        'Rank_PROMETHEE': promethee_result['Posição'].values,
        'Phi_Net': promethee_result['Phi_Net'].values
    })
    
    print("\nTop 5 por AHP:")
    print(comparison.nsmallest(5, 'Rank_AHP')[['Solução', 'f1 (km)', 'f2 (equipes)', 
                                                 'Rank_AHP', 'Score_AHP']])
    
    print("\nTop 5 por PROMETHEE:")
    print(comparison.nsmallest(5, 'Rank_PROMETHEE')[['Solução', 'f1 (km)', 'f2 (equipes)', 
                                                       'Rank_PROMETHEE', 'Phi_Net']])
    
    # 6. Escolha final (desempate se necessário)
    print("\n--- Etapa 6: Escolha da Solução Final ---")
    
    best_ahp_idx = ahp_result['Posição'].idxmin()
    best_promethee_idx = promethee_result['Posição'].idxmin()
    
    if best_ahp_idx == best_promethee_idx:
        print(f"✓ Consenso! Ambos os métodos escolheram a solução {best_ahp_idx}")
        chosen_idx = best_ahp_idx
    else:
        print(f"⚠️ Divergência: AHP escolheu solução {best_ahp_idx}, "
              f"PROMETHEE escolheu {best_promethee_idx}")
        print("\nCritério de desempate: Solução com melhor posição média entre os métodos")
        
        avg_ranks = (ahp_result['Posição'] + promethee_result['Posição']) / 2
        chosen_idx = avg_ranks.idxmin()
        print(f"   → Solução escolhida: {chosen_idx} (rank médio: {avg_ranks[chosen_idx]:.1f})")
    
    chosen_solution = enriched.iloc[chosen_idx]
    
    print(f"\n{'='*60}")
    print("🏆 SOLUÇÃO FINAL ESCOLHIDA")
    print(f"{'='*60}")
    print(f"Índice: {chosen_idx}")
    print(f"f1 - Distância Total: {chosen_solution['f1_distancia']:.2f} km")
    print(f"f2 - Número de Equipes: {int(chosen_solution['f2_equipes'])}")
    print(f"Robustez: {chosen_solution['robustez']:.3f}")
    print(f"Confiabilidade: {chosen_solution['confiabilidade']:.3f}")
    print(f"Risco Operacional: {chosen_solution['risco_operacional']:.2f}/10")
    print(f"Flexibilidade: {chosen_solution['flexibilidade']:.3f}")
    if 'peso_w1' in chosen_solution:
        print(f"Pesos Originais: w1={chosen_solution['peso_w1']:.3f}, w2={chosen_solution['peso_w2']:.3f}")
    
    return {
        'consolidator': consolidator,
        'enriched_solutions': enriched,
        'ahp_result': ahp_result,
        'promethee_result': promethee_result,
        'comparison': comparison,
        'chosen_idx': chosen_idx,
        'chosen_solution': chosen_solution
    }


if __name__ == "__main__":
    results = run_multicriteria_decision(
        results_dir='result',
        method='soma_ponderada',
        max_solutions=20
    )

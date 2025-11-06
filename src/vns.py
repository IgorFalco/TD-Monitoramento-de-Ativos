import random
import time
from neighborhoods import (
    NEIGHBORHOODS, generate_complete_neighborhood
)
import numpy as np
from constraints import constraints
from obj_functions import objective_function_1, objective_function_2, objective_function_weighted

class VNS:
    """Variable Neighborhood Search para otimização de soluções."""
    
    def __init__(self, dist_bases_assets):
        self.dist_bases_assets = dist_bases_assets
        self.neighborhoods = NEIGHBORHOODS
        self.history = []
    
    def local_search(self, solution, objective='f1', max_iter=1000, w1=None, w2=None, epsilon=None, f1_max=None, f2_max=8, f1_min=None, f2_min=None):
        """
        Busca local usando First Improvement.
        
        Args:
            solution: Solução atual
            objective: 'f1', 'f2', ou 'weighted'
            max_iter: Número máximo de iterações
            w1, w2: Pesos para função objetivo ponderada (se objective='weighted')
            epsilon: Limite para f2 no método epsilon-constraint (se fornecido)
            f1_max, f2_max: Valores para normalização na função ponderada
            f1_min, f2_min: Valores mínimos para normalização (min–max), opcionais
        """
        best_solution = self._copy_solution(solution)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            
            for neighborhood_func in self.neighborhoods:
                neighbors = generate_complete_neighborhood(
                    best_solution, neighborhood_func, self.dist_bases_assets, max_neighbors=20
                )
                
                for neighbor in neighbors:
                    # Verifica restrição epsilon se fornecida
                    if epsilon is not None:
                        objective_function_2(neighbor)
                        if neighbor['f2'] > epsilon:
                            continue
                    
                    if self._is_better(neighbor, best_solution, objective, w1, w2, f1_max, f2_max, f1_min, f2_min):
                        best_solution = self._copy_solution(neighbor)
                        improved = True
                        break  # First Improvement
                
                if improved:
                    break
        
        return best_solution
    
    def shake(self, solution, k):
        """
        Fase de shake: reorganiza equipes trocando-as para outras bases.
        Quanto maior o k, mais equipes são trocadas.
        Não cria nem remove equipes, apenas reorganiza as existentes.
        """

        perturbed = self._copy_solution(solution)
        num_bases_y, num_teams = perturbed['y'].shape
        
        # Encontra equipes ativas
        active_teams = []
        for team in range(num_teams):
            if np.sum(perturbed['y'][:, team]) > 0:
                active_teams.append(team)
        
        if len(active_teams) < 2:
            return perturbed
        
        # Define quantas equipes trocar baseado no k
        # k=1: 1 equipe, k=2: 2 equipes, ..., até no máximo todas as equipes
        num_teams_to_swap = min(k, len(active_teams))
        
        # Seleciona quais equipes serão reorganizadas
        teams_to_swap = random.sample(active_teams, num_teams_to_swap)
        
        # Pega as bases atuais dessas equipes
        current_bases = []
        for team_idx in teams_to_swap:
            current_base = np.where(perturbed['y'][:, team_idx] == 1)[0][0]
            current_bases.append(current_base)
        
        # Embaralha as bases entre as equipes selecionadas
        shuffled_bases = current_bases.copy()
        random.shuffle(shuffled_bases)
        
        # Reatribui as bases embaralhadas às equipes
        for i, team_idx in enumerate(teams_to_swap):
            old_base = current_bases[i]
            new_base = shuffled_bases[i]
            
            if old_base != new_base:
                # Move equipe para nova base
                perturbed['y'][old_base, team_idx] = 0
                perturbed['y'][new_base, team_idx] = 1
                
                # Move todos os ativos da equipe para a nova base
                team_assets = np.where(perturbed['h'][:, team_idx] == 1)[0]
                for asset in team_assets:
                    perturbed['x'][asset, old_base] = 0
                    perturbed['x'][asset, new_base] = 1
        
        # Valida e recalcula objetivos
        try:
            valid_ok = True
            for constraint in constraints:
                if not constraint(perturbed):
                    valid_ok = False
                    break
            
            if valid_ok:
                objective_function_1(perturbed, self.dist_bases_assets)
                objective_function_2(perturbed)
                return perturbed
            else:
                # Se inválida, retorna a solução original
                return solution
                
        except Exception:
            return solution
    

    
    def execute(self, initial_solution, objective='f1', max_iter=100, max_time=300, 
                k_max=5, verbose=True, w1=None, w2=None, epsilon=None, f1_max=None, f2_max=8, f1_min=None, f2_min=None):
        """
        Executa o algoritmo VNS completo.
        
        Args:
            initial_solution: Solução inicial
            objective: 'f1' para distância, 'f2' para número de equipes, 'weighted' para soma ponderada
            max_iter: Máximo de iterações
            max_time: Tempo limite em segundos
            k_max: Número máximo de vizinhanças para shake
            verbose: Se deve imprimir informações detalhadas
            w1, w2: Pesos para função objetivo ponderada (necessário se objective='weighted')
            epsilon: Limite para f2 no método epsilon-constraint
            f1_max, f2_max: Valores para normalização na função ponderada
            f1_min, f2_min: Valores mínimos para normalização (min–max), opcionais
        """
        
        start_time = time.time()
        best_solution = self._copy_solution(initial_solution)
        current_solution = self._copy_solution(initial_solution)
        
        iteration = 0
        k = 1
        
        if verbose:
            print(f"\n=== INICIANDO VNS ===")
            print(f"objetivo: {objective}")
            if objective == 'weighted':
                print(f"Pesos: w1={w1}, w2={w2}")
                print(f"Normalização: f1_max={f1_max}, f2_max={f2_max}")
            if epsilon is not None:
                print(f"Restrição: f2 <= {epsilon}")
            
            if objective == 'weighted':
                valor_obj = objective_function_weighted(best_solution, self.dist_bases_assets, w1, w2, f1_max, f2_max, f1_min, f2_min)
                print(f"Solução inicial: F(x)={valor_obj:.4f} (f1={best_solution['f1']:.2f}, f2={best_solution['f2']})")
            else:
                print(f"Solução inicial: {objective}={best_solution[objective]:.2f}")
            print(f"Limites: {max_iter} iterações, {max_time}s")
        
        while (iteration < max_iter and 
               (time.time() - start_time) < max_time and 
               k <= k_max):
            
            iteration += 1
            
            if verbose and iteration % 10 == 0:
                print(f"\n--- Iteração {iteration} (k={k}) ---")
            
            # FASE 1: Shake - gera uma solução na k-ésima vizinhança
            solution_shake = self.shake(current_solution, k)
            
            # FASE 2: Busca local
            solution_local = self.local_search(solution_shake, objective, max_iter=100, 
                                              w1=w1, w2=w2, epsilon=epsilon, f1_max=f1_max, f2_max=f2_max, f1_min=f1_min, f2_min=f2_min)
            
            # FASE 3: Aceita ou rejeita
            if self._is_better(solution_local, best_solution, objective, w1, w2, f1_max, f2_max, f1_min, f2_min):
                best_solution = self._copy_solution(solution_local)
                current_solution = self._copy_solution(solution_local)
                
                self.history.append({
                    'iteracao': iteration,
                    'tempo': time.time() - start_time,
                    'f1': best_solution['f1'],
                    'f2': best_solution['f2'],
                    'melhoria': True
                })
                
                k = 1
                
                if verbose:
                    if objective == 'weighted':
                        valor_obj = objective_function_weighted(best_solution, self.dist_bases_assets, w1, w2, f1_max, f2_max, f1_min, f2_min)
                        print(f"  *** NOVA MELHOR: F(x)={valor_obj:.4f} (f1={best_solution['f1']:.2f}, f2={best_solution['f2']}) ***")
                    else:
                        print(f"  *** NOVA MELHOR: {objective}={best_solution[objective]:.2f} ***")
            
            else:
                current_solution = self._copy_solution(solution_shake)
                k += 1
                
                self.history.append({
                    'iteracao': iteration,
                    'tempo': time.time() - start_time,
                    'f1': best_solution['f1'],
                    'f2': best_solution['f2'],
                    'melhoria': False
                })
        
        tempo_total = time.time() - start_time
        
        if verbose:
            print(f"\n=== VNS FINALIZADO ===")
            print(f"Iterações: {iteration}")
            print(f"Tempo total: {tempo_total:.2f}s")
            if objective == 'weighted':
                valor_obj = objective_function_weighted(best_solution, self.dist_bases_assets, w1, w2, f1_max, f2_max, f1_min, f2_min)
                print(f"Melhor solução: F(x)={valor_obj:.4f} (f1={best_solution['f1']:.2f}, f2={best_solution['f2']})")
            else:
                print(f"Melhor solução: {objective}={best_solution[objective]:.2f}")
            
            if objective == 'f1':
                melhoria_percentual = ((initial_solution['f1'] - best_solution['f1']) / 
                                     initial_solution['f1']) * 100
                print(f"Melhoria em f1: {melhoria_percentual:.2f}%")
            elif objective == 'f2':
                teams_reduction = initial_solution['f2'] - best_solution['f2']
                print(f"Redução de equipes: {teams_reduction}")
        
        return best_solution
    
    def _is_better(self, solution1, solution2, objective, w1=None, w2=None, f1_max=None, f2_max=8, f1_min=None, f2_min=None):
        """Verifica se solution1 é melhor que solution2."""
        if objective == 'f1':
            return solution1['f1'] < solution2['f1']
        elif objective == 'f2':
            return solution1['f2'] < solution2['f2']
        elif objective == 'weighted':
            # Calcula função objetivo ponderada normalizada para ambas soluções
            f_sol1 = objective_function_weighted(solution1, self.dist_bases_assets, w1, w2, f1_max, f2_max, f1_min, f2_min)
            f_sol2 = objective_function_weighted(solution2, self.dist_bases_assets, w1, w2, f1_max, f2_max, f1_min, f2_min)
            return f_sol1 < f_sol2
        return False
        
    def _copy_solution(self, solution):
        return {
            'x': solution['x'].copy(),
            'y': solution['y'].copy(), 
            'h': solution['h'].copy(),
            'f1': solution['f1'],
            'f2': solution['f2']
        }
    
    def print_history(self):
        print("\n=== HISTÓRICO DE MELHORIAS ===")
        for record in self.history:
            status = "✓" if record['melhoria'] else "○"
            print(f"{status} Iter {record['iteracao']:3d} | "
                  f"Tempo: {record['tempo']:6.1f}s | "
                  f"f1: {record['f1']:7.2f} | "
                  f"f2: {int(record['f2']):2d}")

def create_vns(dist_bases_assets):
    """Factory function para criar instância do VNS."""
    return VNS(dist_bases_assets)
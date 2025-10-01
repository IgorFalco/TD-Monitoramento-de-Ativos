import random
import time
from neighborhoods import (
    NEIGHBORHOODS, generate_complete_neighborhood
)
import numpy as np
from constraints import constraints
from obj_functions import objective_function_1, objective_function_2

class VNS:
    """Variable Neighborhood Search para otimização de soluções."""
    
    def __init__(self, dist_bases_assets):
        self.dist_bases_assets = dist_bases_assets
        self.neighborhoods = NEIGHBORHOODS
        self.history = []
    
    def local_search(self, solution, objective='f1', max_iter=1000):
        """Busca local usando First Improvement."""
        melhor_solution = self._copy_solution(solution)
        melhorou = True
        iteracoes = 0
        
        print(f"    Iniciando busca local (objective: {objective})")
        
        while melhorou and iteracoes < max_iter:
            melhorou = False
            iteracoes += 1
            
            for neighborhood_func in self.neighborhoods:
                neighbors = generate_complete_neighborhood(melhor_solution, neighborhood_func, 
                                                         self.dist_bases_assets, max_neighbors=20)
                
                for neighbor in neighbors:
                    if self._is_better(neighbor, melhor_solution, objective):
                        melhor_solution = self._copy_solution(neighbor)
                        melhorou = True
                        print(f"      Melhoria: {objective}={melhor_solution[objective]:.2f}")
                        break
                
                if melhorou:
                    break
        
        print(f"    Busca local finalizada: {iteracoes} iterações")
        return melhor_solution
    
    def shake(self, solution, k):
        """
        Fase de shake: reorganiza equipes trocando-as para outras bases.
        Quanto maior o k, mais equipes são trocadas.
        Não cria nem remove equipes, apenas reorganiza as existentes.
        """

        perturbed = self._copy_solution(solution)
        num_bases_y, num_equipes = perturbed['y'].shape
        
        # Encontra equipes ativas
        equipes_ativas = []
        for team in range(num_equipes):
            if np.sum(perturbed['y'][:, team]) > 0:
                equipes_ativas.append(team)
        
        if len(equipes_ativas) < 2:
            return perturbed
        
        # Define quantas equipes trocar baseado no k
        # k=1: 1 equipe, k=2: 2 equipes, ..., até no máximo todas as equipes
        num_equipes_trocar = min(k, len(equipes_ativas))
        
        # Seleciona quais equipes serão reorganizadas
        equipes_para_trocar = random.sample(equipes_ativas, num_equipes_trocar)
        
        # Pega as bases atuais dessas equipes
        bases_atuais = []
        for equipe in equipes_para_trocar:
            base_atual = np.where(perturbed['y'][:, equipe] == 1)[0][0]
            bases_atuais.append(base_atual)
        
        # Embaralha as bases entre as equipes selecionadas
        bases_embaralhadas = bases_atuais.copy()
        random.shuffle(bases_embaralhadas)
        
        # Reatribui as bases embaralhadas às equipes
        for i, equipe in enumerate(equipes_para_trocar):
            base_antiga = bases_atuais[i]
            nova_base = bases_embaralhadas[i]
            
            if base_antiga != nova_base:
                # Move equipe para nova base
                perturbed['y'][base_antiga, equipe] = 0
                perturbed['y'][nova_base, equipe] = 1
                
                # Move todos os ativos da equipe para a nova base
                ativos_equipe = np.where(perturbed['h'][:, equipe] == 1)[0]
                for ativo in ativos_equipe:
                    perturbed['x'][ativo, base_antiga] = 0
                    perturbed['x'][ativo, nova_base] = 1
        
        # Valida e recalcula objetivos
        try:
            valid = True
            for constraint in constraints:
                if not constraint(perturbed):
                    valid = False
                    break
            
            if valid:
                objective_function_1(perturbed, self.dist_bases_assets)
                objective_function_2(perturbed)
                return perturbed
            else:
                # Se inválida, retorna a solução original
                return solution
                
        except Exception:
            return solution
    

    
    def execute(self, solution_inicial, objective='f1', max_iter=100, max_time=300, 
                k_max=5, verbose=True):
        """
        Executa o algoritmo VNS completo.
        
        Args:
            solution_inicial: Solução inicial
            objective: 'f1' para distância, 'f2' para número de equipes
            max_iter: Máximo de iterações
            max_time: Tempo limite em segundos
            k_max: Número máximo de vizinhanças para shake
            verbose: Se deve imprimir informações detalhadas
        """
        
        start_time = time.time()
        melhor_solution = self._copy_solution(solution_inicial)
        current_solution = self._copy_solution(solution_inicial)
        
        iteracao = 0
        k = 1
        
        if verbose:
            print(f"\n=== INICIANDO VNS ===")
            print(f"objetivo: {objective}")
            print(f"Solução inicial: {objective}={melhor_solution[objective]:.2f}")
            print(f"Limites: {max_iter} iterações, {max_time}s")
        
        while (iteracao < max_iter and 
               (time.time() - start_time) < max_time and 
               k <= k_max):
            
            iteracao += 1
            
            if verbose:
                print(f"\n--- Iteração {iteracao} (k={k}) ---")
            
            # FASE 1: Shake - gera uma solução na k-ésima vizinhança
            solution_shake = self.shake(current_solution, k)
            
            if verbose:
                print(f"  Shake: {objective}={solution_shake[objective]:.2f}")
            
            # FASE 2: Busca local
            solution_local = self.local_search(solution_shake, objective)
            
            # FASE 3: Aceita ou rejeita
            if self._is_better(solution_local, melhor_solution, objective):
                melhor_solution = self._copy_solution(solution_local)
                current_solution = self._copy_solution(solution_local)
                
                self.history.append({
                    'iteracao': iteracao,
                    'tempo': time.time() - start_time,
                    'f1': melhor_solution['f1'],
                    'f2': melhor_solution['f2'],
                    'melhoria': True
                })
                
                k = 1
                
                if verbose:
                    print(f"  *** NOVA MELHOR: {objective}={melhor_solution[objective]:.2f} ***")
            
            else:
                current_solution = self._copy_solution(solution_shake)
                k += 1
                
                self.history.append({
                    'iteracao': iteracao,
                    'tempo': time.time() - start_time,
                    'f1': melhor_solution['f1'],
                    'f2': melhor_solution['f2'],
                    'melhoria': False
                })
                
                if verbose:
                    print(f"  Sem melhoria, k={k}")
        
        tempo_total = time.time() - start_time
        
        if verbose:
            print(f"\n=== VNS FINALIZADO ===")
            print(f"Iterações: {iteracao}")
            print(f"Tempo total: {tempo_total:.2f}s")
            print(f"Melhor solução: {objective}={melhor_solution[objective]:.2f}")
            
            if objective == 'f1':
                melhoria_percentual = ((solution_inicial['f1'] - melhor_solution['f1']) / 
                                     solution_inicial['f1']) * 100
                print(f"Melhoria em f1: {melhoria_percentual:.2f}%")
            else:
                melhoria_equipes = solution_inicial['f2'] - melhor_solution['f2']
                print(f"Redução de equipes: {melhoria_equipes}")
        
        return melhor_solution
    
    def _is_better(self, solution1, solution2, objective):
        if objective == 'f1':
            return solution1['f1'] < solution2['f1']
        if objective == 'f2':
            return solution1['f2'] < solution2['f2']
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
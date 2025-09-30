import random
import time
from neighborhoods import (
    NEIGHBORHOODS, generate_complete_neighborhood
)

class VNS:
    """Variable Neighborhood Search para otimização de soluções."""
    
    def __init__(self, dist_bases_assets):
        self.dist_bases_assets = dist_bases_assets
        self.neighborhoods = NEIGHBORHOODS
        self.history = []
    
    def busca_local(self, solution, objetivo='f1', max_iter=1000):
        """Busca local usando First Improvement."""
        melhor_solution = self._copy_solution(solution)
        melhorou = True
        iteracoes = 0
        
        print(f"    Iniciando busca local (objetivo: {objetivo})")
        
        while melhorou and iteracoes < max_iter:
            melhorou = False
            iteracoes += 1
            
            for neighborhood_func in self.neighborhoods:
                neighbors = generate_complete_neighborhood(melhor_solution, neighborhood_func, 
                                                         self.dist_bases_assets, max_neighbors=20)
                
                for neighbor in neighbors:
                    if self._is_better(neighbor, melhor_solution, objetivo):
                        melhor_solution = self._copy_solution(neighbor)
                        melhorou = True
                        print(f"      Melhoria: {objetivo}={melhor_solution[objetivo]:.2f}")
                        break
                
                if melhorou:
                    break
        
        print(f"    Busca local finalizada: {iteracoes} iterações")
        return melhor_solution
    
    def shake(self, solution, k):
        """Fase de shake: aplica k movimentos aleatórios."""
        current_solution = self._copy_solution(solution)
        
        for _ in range(k):
            neighborhood_func = random.choice(self.neighborhoods)
            new_neighbor = neighborhood_func(current_solution, self.dist_bases_assets)
            
            if new_neighbor is not None:
                current_solution = new_neighbor
        
        return current_solution
    
    def otimizar(self, solution_inicial, objetivo='f1', max_iter=100, max_time=300, 
                k_max=5, verbose=True):
        """
        Executa o algoritmo VNS completo.
        
        Args:
            solution_inicial: Solução inicial
            objetivo: 'f1' para distância, 'f2' para número de equipes
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
            print(f"Objetivo: {objetivo}")
            print(f"Solução inicial: {objetivo}={melhor_solution[objetivo]:.2f}")
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
                print(f"  Shake: {objetivo}={solution_shake[objetivo]:.2f}")
            
            # FASE 2: Busca local
            solution_local = self.busca_local(solution_shake, objetivo)
            
            # FASE 3: Aceita ou rejeita
            if self._is_better(solution_local, melhor_solution, objetivo):
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
                    print(f"  *** NOVA MELHOR: {objetivo}={melhor_solution[objetivo]:.2f} ***")
            
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
            print(f"Melhor solução: {objetivo}={melhor_solution[objetivo]:.2f}")
            
            if objetivo == 'f1':
                melhoria_percentual = ((solution_inicial['f1'] - melhor_solution['f1']) / 
                                     solution_inicial['f1']) * 100
                print(f"Melhoria em f1: {melhoria_percentual:.2f}%")
            else:
                melhoria_equipes = solution_inicial['f2'] - melhor_solution['f2']
                print(f"Redução de equipes: {melhoria_equipes}")
        
        return melhor_solution
    
    def _is_better(self, solution1, solution2, objetivo):
        if objetivo == 'f1':
            return solution1['f1'] < solution2['f1']
        if objetivo == 'f2':
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
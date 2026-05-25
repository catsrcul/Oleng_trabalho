"""
GCMMA Otimizador Simplificado
==============================

Uso direto na main:

    from gcmma_simples import Otimizador
    
    otimizador = Otimizador(SeuProblema, "config.yaml")
    x_otimo = otimizador.otimizar()
    
    # Acessar resultados
    print(f"f(x) = {otimizador.f_otimo}")
    print(f"Violação = {otimizador.violacao}")
    otimizador.graficar()
"""

import numpy as np
# from scipy.optimize import minimize, Bounds
from abc import ABC, abstractmethod
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class ProblemaOtimizacao(ABC):
    """Interface que todos os problemas devem implementar."""
    
    @abstractmethod
    def objetivo(self, x: np.ndarray) -> float:
        """Retorna f(x) — MAXIMIZAR se valor positivo, MINIMIZAR se negativo."""
        pass
    
    @abstractmethod
    def restricoes(self, x: np.ndarray) -> Dict[str, float]:
        """
        Retorna {'g1': ..., 'g2': ..., ...}
        Convenção: g(x) <= 0 é VIÁVEL
        """
        pass
    
    @abstractmethod
    def limites(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna (x_min, x_max)."""
        pass
    
    @abstractmethod
    def inicial(self) -> np.ndarray:
        """Retorna design inicial."""
        pass
    
    @property
    @abstractmethod
    def n_vars(self) -> int:
        """Número de variáveis."""
        pass
    
    @property
    @abstractmethod
    def n_restricoes(self) -> int:
        """Número de restrições."""
        pass


class Otimizador:
    """Otimizador GCMMA para uso direto."""
    
    def __init__(self, problema_class, config_path=None, 
                 move=0.1, rho=50, tol=1e-4, max_iter=150, verbose=True):
        """
        Args:
            problema_class: Classe que herda ProblemaOtimizacao
            config_path: Caminho para config.yaml (se necessário)
            move: Tamanho do movimento das assimetotes (0.01-0.2)
            rho: Penalidade do subproblema (10-100)
            tol: Tolerância de convergência
            max_iter: Máximo de iterações
            verbose: Imprimir progresso
        """
        self.problema = problema_class(config_path) if config_path else problema_class()
        self.move = move
        self.rho = rho
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        
        # Resultados
        self.x_otimo = None
        self.f_otimo = None
        self.restricoes_otimas = None
        self.violacao = None
        self.n_iter = None
        self.historico = {'f': [], 'violacao': [], 'passo': []}

    def otimizar(self) -> np.ndarray:
        x = self.problema.inicial()
        xmin, xmax = self.problema.limites()

        # Inicializar assimetotes e histórico (Zillober Sec 5)
        L = np.maximum(xmin - 0.1 * (xmax - xmin), xmin - 1.0)
        U = np.minimum(xmax + 0.1 * (xmax - xmin), xmax + 1.0)
        self.L_prev = L.copy()
        self.U_prev = U.copy()
        self.x_prev1 = x.copy()
        self.x_prev2 = x.copy()

        lam = np.zeros(self.problema.n_restricoes)

        for k in range(self.max_iter):
            # 1. Avaliar problema original
            f = self.problema.objetivo(x)
            g_dict = self.problema.restricoes(x)
            g = np.array([g_dict[f'g{i + 1}'] for i in range(self.problema.n_restricoes)])
            violacao = np.maximum(0, g).sum()

            # 2. Resolver Subproblema (Dual)
            x_sub, passo, lam_sub = self._resolver_subproblema(x, L, U, g, lam, xmin, xmax)

            # 3. Atualizar variáveis (MMA direto)
            x_novo = np.clip(x_sub, xmin, xmax)
            lam = np.maximum(0, lam_sub)

            L, U = self._atualizar_assimetotes(x, x_novo, xmin, xmax, k)

            # Guardar histórico para assimetotes
            self.x_prev2 = self.x_prev1.copy()
            self.x_prev1 = x.copy()
            self.L_prev = L.copy()
            self.U_prev = U.copy()

            # 5. Verificar convergência
            x_change = np.linalg.norm(x_novo - x) / (np.linalg.norm(x) + 1e-10)
            x = x_novo

            self.historico['f'].append(f)
            self.historico['violacao'].append(violacao)

            if self.verbose:
                print(f"Iter {k + 1:<4} | f(x): {f:<10.4e} | Viol: {violacao:<8.2e} | Passo: {passo:<6.3e}")

            if x_change < self.tol and violacao < 1e-3:
                if self.verbose: print("✓ Convergência alcançada.")
                break

        # Guardar resultados finais
        self.x_otimo = x
        self.f_otimo = self.problema.objetivo(x)
        self.restricoes_otimas = self.problema.restricoes(x)
        self.violacao = max([0] + [max(0, v) for v in self.restricoes_otimas.values()])
        self.n_iter = k + 1

        return x

    def _resolver_subproblema(self, x_k, L, U, g, lam, xmin, xmax):
        """Resolve subproblema com otimizador dual customizado e variáveis artificiais."""
        n_vars = len(x_k)
        n_restr = len(g)

        eps = 1e-8  # Era 1e-6
        grad_f = np.zeros(n_vars)
        grad_g = np.zeros((n_restr, n_vars))

        f_k = self.problema.objetivo(x_k)
        g_k = np.array([self.problema.restricoes(x_k)[f'g{i + 1}'] for i in range(n_restr)])

        for i in range(n_vars):
            x_pert = x_k.copy()
            x_pert[i] += eps
            grad_f[i] = (self.problema.objetivo(x_pert) - f_k) / eps
            g_pert = np.array([self.problema.restricoes(x_pert)[f'g{i + 1}'] for i in range(n_restr)])
            grad_g[:, i] = (g_pert - g_k) / eps

        epsilon_convex = 1e-4  # Era 1e-3

        p0 = np.maximum(0, grad_f) * (U - x_k) ** 2 + epsilon_convex * (xmax - xmin) ** 2
        q0 = np.maximum(0, -grad_f) * (x_k - L) ** 2 + epsilon_convex * (xmax - xmin) ** 2

        P = np.zeros((n_restr, n_vars))
        Q = np.zeros((n_restr, n_vars))
        for j in range(n_restr):
            P[j, :] = np.maximum(0, grad_g[j, :]) * (U - x_k) ** 2
            Q[j, :] = np.maximum(0, -grad_g[j, :]) * (x_k - L) ** 2

        move_limit = self.move
        x_min_lim = np.maximum(xmin, x_k - move_limit * (x_k - L))
        x_max_lim = np.minimum(xmax, x_k + move_limit * (U - x_k))

        # Variáveis artificiais (Zillober Sec 4)
        M_violados = g_k > 0  # Identificar restrições violadas
        rho_art = 1000.0  # Penalidade alta para forçar viabilidade
        mu = np.zeros(n_restr)

        lambda_val = np.maximum(0, lam.copy())
        lr = 0.2  # Era 0.05

        for dual_iter in range(200):  # Era 50
            # 1. Primal x
            P_lambda = p0 + np.dot(lambda_val, P)
            Q_lambda = q0 + np.dot(lambda_val, Q)

            P_lambda = np.maximum(P_lambda, 1e-15)
            Q_lambda = np.maximum(Q_lambda, 1e-15)

            x_novo = (U * np.sqrt(Q_lambda) + L * np.sqrt(P_lambda)) / (np.sqrt(P_lambda) + np.sqrt(Q_lambda))
            x_novo = np.clip(x_novo, x_min_lim, x_max_lim)

            # 2. Primal mu (Variáveis artificiais)
            for j in range(n_restr):
                if M_violados[j]:
                    # Formula exata estacionariedade
                    mu_calc = (lambda_val[j] * g_k[j]) / rho_art
                    mu[j] = np.clip(mu_calc, 0.0, 1.0)

            # 3. Gradiente dual
            g_tilde = np.zeros(n_restr)
            for j in range(n_restr):
                g_aprox = g_k[j] + np.sum(
                    P[j, :] / (U - x_novo) + Q[j, :] / (x_novo - L) - P[j, :] / (U - x_k) - Q[j, :] / (x_k - L))
                if M_violados[j]:
                    g_aprox -= mu[j] * g_k[j]  # Relaxar restrição
                g_tilde[j] = g_aprox

            # 4. Atualizar lambda
            lambda_val = np.maximum(0, lambda_val + lr * g_tilde)

        passo = np.linalg.norm(x_novo - x_k)
        return x_novo, passo, lambda_val


    def graficar(self):
        """Plota histórico de convergência."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.historico['f'], 'b-o')
        axes[0].set_xlabel('Iteração')
        axes[0].set_ylabel('f(x)')
        axes[0].set_title('Função Objetivo')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].semilogy(self.historico['violacao'], 'r-o')
        axes[1].set_xlabel('Iteração')
        axes[1].set_ylabel('Violação')
        axes[1].set_title('Violação de Restrição')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def relatorio(self, x_nomes=None):
        """Imprime relatório dos resultados."""
        print(f"\n{'='*70}")
        print(f"RESULTADOS DA OTIMIZAÇÃO")
        print(f"{'='*70}")
        
        print(f"\nDesign Ótimo:")
        if x_nomes:
            for i, nome in enumerate (x_nomes):
                print(f"  {nome:<20} = {self.x_otimo[i]:.6e}")
        else:
            for i, val in enumerate(self.x_otimo):
                print(f"  x[{i}]                   = {val:.6e}")
        
        print(f"\nObjetivo:")
        print(f"  f(x*)                = {self.f_otimo:.6e}")
        
        print(f"\nRestrições:")
        for i in range(1, self.problema.n_restricoes + 1):
            g_val = self.restricoes_otimas[f'g{i}']
            status = "✓ OK" if g_val <= 1e-3 else "✗ VIOLADA"
            print(f"  g{i}                    = {g_val:>12.3e}  {status}")
        
        print(f"\nConvergência:")
        print(f"  Iterações:           {self.n_iter}")
        print(f"  Violação máxima:     {self.violacao:.3e}")
        print(f"{'='*70}\n")

    def _lagrangiano_aumentado(self, x, f_val, g_val, lam, r_penalty):
        """Calcula função Lagrange aumentada (Zillober Def 3.1)."""
        phi = f_val
        for j in range(len(g_val)):
            if -lam[j] / r_penalty <= g_val[j]:
                phi += lam[j] * g_val[j] + 0.5 * r_penalty * g_val[j] ** 2
            else:
                phi += -0.5 * (lam[j] ** 2) / r_penalty
        return phi

    def _atualizar_assimetotes(self, x_k, x_novo, xmin, xmax, k):
        """Atualiza assimetotes segundo Zillober 1993."""
        L = np.zeros_like(x_k)
        U = np.zeros_like(x_k)
        t = 0.7  # Parâmetro Zillober

        if k < 2:
            L = xmin - 0.1 * (xmax - xmin)
            U = xmax + 0.1 * (xmax - xmin)
        else:
            for i in range(len(x_k)):
                sign_k1 = np.sign(x_k[i] - self.x_prev1[i])
                sign_k2 = np.sign(self.x_prev1[i] - self.x_prev2[i])

                if sign_k1 * sign_k2 > 0:
                    L[i] = x_k[i] - (self.x_prev1[i] - self.L_prev[i]) / t
                    U[i] = x_k[i] + (self.U_prev[i] - self.x_prev1[i]) / t
                elif sign_k1 * sign_k2 < 0:
                    L[i] = x_k[i] - t * (self.x_prev1[i] - self.L_prev[i])
                    U[i] = x_k[i] + t * (self.U_prev[i] - self.x_prev1[i])
                else:
                    L[i] = x_k[i] - (self.x_prev1[i] - self.L_prev[i])
                    U[i] = x_k[i] + (self.U_prev[i] - self.x_prev1[i])

        # Prevenir divisão por zero e limites infinitos
        min_dist = 1e-5 * (xmax - xmin)
        L = np.maximum(L, x_k - 10 * (xmax - xmin))
        L = np.minimum(L, x_k - min_dist)

        U = np.minimum(U, x_k + 10 * (xmax - xmin))
        U = np.maximum(U, x_k + min_dist)

        return L, U
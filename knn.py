"""
Implementação base do algoritmo k-Nearest Neighbors (KNN) para classificação.

Um objeto novo é classificado pela votação dos seus k vizinhos mais próximos:
ele recebe a classe mais comum entre esses k vizinhos. Se k = 1, recebe a
classe do único vizinho mais próximo.

Distâncias disponíveis (para N atributos, i = 1..N):
    Euclidiana: d(p, q) = sqrt( sum (p_i - q_i)^2 )
    Manhattan:  d(p, q) = sum |p_i - q_i|
"""

from collections import Counter
from math import sqrt
from typing import Callable, Hashable, Sequence

Ponto = Sequence[float]
Classe = Hashable


def distancia_euclidiana(p: Ponto, q: Ponto) -> float:
    """Raiz da soma dos quadrados das diferenças entre os atributos."""
    _validar_dimensoes(p, q)
    return sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))


def distancia_manhattan(p: Ponto, q: Ponto) -> float:
    """Soma dos módulos das diferenças entre os atributos."""
    _validar_dimensoes(p, q)
    return sum(abs(pi - qi) for pi, qi in zip(p, q))


METRICAS: dict[str, Callable[[Ponto, Ponto], float]] = {
    "euclidiana": distancia_euclidiana,
    "manhattan": distancia_manhattan,
}


def _validar_dimensoes(p: Ponto, q: Ponto) -> None:
    if len(p) != len(q):
        raise ValueError(
            f"Os pontos têm números de atributos diferentes: {len(p)} e {len(q)}."
        )


class KNN:
    """
    Classificador k-NN.

    Uso:
        modelo = KNN(k=3, metrica="euclidiana")
        modelo.treinar(X, y)
        classe = modelo.classificar([3, 4.5])
    """

    def __init__(self, k: int = 3, metrica: str = "euclidiana") -> None:
        if not isinstance(k, int) or k < 1:
            raise ValueError("k deve ser um número inteiro positivo.")
        if metrica not in METRICAS:
            raise ValueError(
                f"Métrica '{metrica}' desconhecida. Use: {', '.join(METRICAS)}."
            )
        self.k = k
        self.metrica = metrica
        self._distancia = METRICAS[metrica]
        self._X: list[Ponto] = []
        self._y: list[Classe] = []

    def treinar(self, X: Sequence[Ponto], y: Sequence[Classe]) -> "KNN":
        """
        O KNN é um algoritmo "preguiçoso" (lazy): treinar é só guardar as
        instâncias conhecidas. Todo o cálculo acontece na classificação.
        """
        if len(X) != len(y):
            raise ValueError("X e y devem ter a mesma quantidade de instâncias.")
        if len(X) == 0:
            raise ValueError("É preciso pelo menos uma instância para treinar.")
        dimensao = len(X[0])
        if any(len(x) != dimensao for x in X):
            raise ValueError("Todas as instâncias devem ter o mesmo número de atributos.")
        if self.k > len(X):
            raise ValueError(
                f"k ({self.k}) não pode ser maior que o número de instâncias ({len(X)})."
            )
        self._X = [list(x) for x in X]
        self._y = list(y)
        return self

    def vizinhos(self, p: Ponto) -> list[tuple[float, Ponto, Classe]]:
        """
        Retorna os k vizinhos mais próximos de p como (distância, ponto, classe),
        ordenados da menor para a maior distância.
        """
        if not self._X:
            raise RuntimeError("O modelo ainda não foi treinado. Chame treinar() antes.")
        distancias = [
            (self._distancia(p, x), x, classe) for x, classe in zip(self._X, self._y)
        ]
        # sort é estável: em distâncias iguais, mantém a ordem dos dados de treino.
        distancias.sort(key=lambda item: item[0])
        return distancias[: self.k]

    def classificar(self, p: Ponto) -> Classe:
        """Classifica p pela classe mais votada entre os k vizinhos."""
        return self.classificar_detalhado(p)["classe"]

    def classificar_detalhado(self, p: Ponto) -> dict:
        """
        Classifica p e devolve também os vizinhos, os votos e se houve empate.

        Desempate: quando duas ou mais classes recebem o mesmo número de votos
        (comum com k par), vence a classe cuja soma das distâncias dos seus
        vizinhos é menor, ou seja, a que está "mais perto" no conjunto.
        """
        vizinhos = self.vizinhos(p)
        votos = Counter(classe for _, _, classe in vizinhos)
        maior_voto = max(votos.values())
        empatadas = [classe for classe, qtd in votos.items() if qtd == maior_voto]

        if len(empatadas) == 1:
            vencedora = empatadas[0]
        else:
            soma_dist = {
                classe: sum(d for d, _, c in vizinhos if c == classe)
                for classe in empatadas
            }
            vencedora = min(empatadas, key=lambda c: soma_dist[c])

        return {
            "classe": vencedora,
            "vizinhos": vizinhos,
            "votos": dict(votos),
            "empate": len(empatadas) > 1,
        }

    def classificar_varios(self, pontos: Sequence[Ponto]) -> list[Classe]:
        return [self.classificar(p) for p in pontos]

    def acuracia(self, X_teste: Sequence[Ponto], y_teste: Sequence[Classe]) -> float:
        """Fração de instâncias de teste classificadas corretamente (0 a 1)."""
        if len(X_teste) != len(y_teste) or not X_teste:
            raise ValueError("X_teste e y_teste devem ter o mesmo tamanho (> 0).")
        acertos = sum(
            1 for p, esperado in zip(X_teste, y_teste) if self.classificar(p) == esperado
        )
        return acertos / len(X_teste)

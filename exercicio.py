"""
Exercício da aula:
Determinar a classificação de uma instância nova, p(3, 4.5), para k = {1, 2, 3, 4},
usando a distância Euclidiana e a distância Manhattan.

Os pontos abaixo são os do gráfico do slide (classes A e B).
Execute com:  python exercicio.py
"""

from knn import KNN

CLASSE_A = [
    (-2, 1), (-1, 4), (0, 0), (1, 8), (2, 6), (3, 9), (4, 5),
    (5, 7), (6, 8), (7, 10), (8, -3), (9, 6), (10, 9),
]
CLASSE_B = [
    (-2, 1), (-1, -3), (0, 6), (1, 1), (2, 4.5), (3, 3), (4, 1),
    (5, 3), (6, 4), (7, 5), (8, 2), (9, 3), (10, 0),
]

X = CLASSE_A + CLASSE_B
y = ["A"] * len(CLASSE_A) + ["B"] * len(CLASSE_B)

P = (3, 4.5)
VALORES_K = [1, 2, 3, 4]


def formatar_ponto(ponto) -> str:
    return "(" + ", ".join(f"{v:g}" for v in ponto) + ")"


def main() -> None:
    print(f"Nova instância p = {formatar_ponto(P)}\n")

    for metrica in ("euclidiana", "manhattan"):
        print("=" * 60)
        print(f"Distância {metrica.capitalize()}")
        print("=" * 60)

        # Mostra os vizinhos mais próximos uma vez (o maior k cobre os demais)
        modelo = KNN(k=max(VALORES_K), metrica=metrica).treinar(X, y)
        print("Vizinhos mais próximos:")
        for posicao, (dist, ponto, classe) in enumerate(modelo.vizinhos(P), start=1):
            print(f"  {posicao}º  {formatar_ponto(ponto):<10} classe {classe}  d = {dist:.4f}")
        print()

        for k in VALORES_K:
            resultado = KNN(k=k, metrica=metrica).treinar(X, y).classificar_detalhado(P)
            votos = ", ".join(f"{c}={n}" for c, n in sorted(resultado["votos"].items()))
            aviso = "  (empate - desempate pela menor soma de distâncias)" if resultado["empate"] else ""
            print(f"  k = {k}: votos [{votos}] -> classe {resultado['classe']}{aviso}")
        print()


if __name__ == "__main__":
    main()

# k-Nearest Neighbors (KNN)

Implementação base do algoritmo KNN para classificação, em Python puro (sem bibliotecas externas).

## Como funciona

Um objeto novo é classificado pela votação dos seus **k vizinhos mais próximos**: ele recebe a classe mais comum entre eles. Com `k = 1`, recebe a classe do único vizinho mais próximo.

Distâncias implementadas, para N atributos (`i = 1..N`):

- **Euclidiana:** `d(p, q) = √ Σ (pᵢ − qᵢ)²`
- **Manhattan:** `d(p, q) = Σ |pᵢ − qᵢ|`

Quando há empate na votação (comum com `k` par), vence a classe cuja soma das distâncias dos seus vizinhos é menor.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `knn.py` | Funções de distância e a classe `KNN` |
| `exercicio.py` | Resolução do exercício da aula: p(3, 4.5), k = {1, 2, 3, 4}, Euclidiana e Manhattan |

## Uso

```python
from knn import KNN

X = [(1, 1), (2, 1), (8, 9), (9, 8)]
y = ["A", "A", "B", "B"]

modelo = KNN(k=3, metrica="euclidiana")   # ou metrica="manhattan"
modelo.treinar(X, y)

modelo.classificar((2, 2))            # 'A'
modelo.classificar_detalhado((2, 2))  # classe, vizinhos, votos e se houve empate
```

## Executar o exercício

```bash
python exercicio.py
```

Requer Python 3.9 ou superior.

## Observação: maldição da dimensionalidade

Para dados com muitas dimensões (mais de ~10), a distância euclidiana perde significado, pois os pontos ficam quase equidistantes da consulta. Nesses casos, aplica-se redução de dimensão (PCA, LDA ou CCA) antes do KNN.

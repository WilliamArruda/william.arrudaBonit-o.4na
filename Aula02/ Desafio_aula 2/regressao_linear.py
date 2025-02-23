# Implementação da Regressão Linear Simples sem bibliotecas externas

# Passo 1: Definir as listas de dados
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Passo 2: Calcular as médias de x e y
n = len(x)  # Número de elementos
media_x = sum(x) / n
media_y = sum(y) / n

# Passo 3: Calcular os somatórios necessários para beta1
soma_num = 0  # Numerador da fórmula de beta1
soma_den = 0  # Denominador da fórmula de beta1

for i in range(n):
    soma_num += (x[i] - media_x) * (y[i] - media_y)
    soma_den += (x[i] - media_x) ** 2

# Passo 4: Calcular os coeficientes beta1 e beta0
beta1 = soma_num / soma_den
beta0 = media_y - beta1 * media_x

# Passo 5: Imprimir os resultados
print(f"Coeficiente beta1 (inclinação): {beta1}")
print(f"Coeficiente beta0 (intercepto): {beta0}")

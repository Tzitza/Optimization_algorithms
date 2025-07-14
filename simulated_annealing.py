import numpy as np

# -------------------------- Ορισμός Συναρτήσεων --------------------------
def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def rastrigin_grad(x):
    return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)

def camel(x):
    return (x[0]**2 + x[1]**2) + (2 * x[0]**2 - 2 * x[0] * x[1] + 3 * x[1]**2)

def camel_grad(x):
    return np.array([4 * x[0] - 2 * x[1], -2 * x[0] + 6 * x[1]])

def shubert(x):
    return np.prod([np.sum([(i + 1) * np.cos((i + 2) * x[j] + i + 1) for i in range(5)]) for j in range(len(x))])

def shubert_grad(x):
    return np.array([np.sum([-(i + 1) * (i + 2) * np.sin((i + 2) * x[j] + i + 1) for i in range(5)]) for j in range(len(x))])

def griewank2(x):
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1

def griewank2_grad(x):
    return (x / 2000) + np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) * np.sin(x / np.sqrt(np.arange(1, len(x) + 1))) / np.sqrt(np.arange(1, len(x) + 1))

def branin(x):
    a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)
    return a * (x[1] - b * x[0]**2 + c * x[0] - r)**2 + s * (1 - t) * np.cos(x[0]) + s

def branin_grad(x):
    a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)
    return np.array([-2 * a * (x[1] - b * x[0]**2 + c * x[0] - r) * (2 * b * x[0] - c) - s * (1 - t) * np.sin(x[0]), 2 * a * (x[1] - b * x[0]**2 + c * x[0] - r)])

# Gradient Descent για τοπική βελτιστοποίηση
def gradient_descent(func, grad_func, x_init, lr=0.01, max_iter=100):
    """Gradient Descent με έλεγχο μείωσης της συνάρτησης κόστους"""
    x = np.array(x_init, dtype=np.float64)
    prev_f = func(x)  # Αρχική τιμή συνάρτησης κόστους
    for _ in range(max_iter):
        grad_val = np.array(grad_func(x))
        x -= lr * grad_val
        new_f = func(x)
        if abs(new_f - prev_f) < 1e-6:  # Σταμάτα αν η αλλαγή στη συνάρτηση είναι πολύ μικρή
            break
        prev_f = new_f
    return x


# Υλοποίηση του Simulated Annealing (μέθοδος Corana)
def simulated_annealing(func, grad, x_init, temp_init=1000, alpha=0.85, max_iter=1000, Ns=20, Nt=100):
    x_current = np.array(x_init, dtype=np.float64)
    best_x = x_current.copy()
    best_f = func(x_current)
    temp = temp_init
    step_size = np.ones_like(x_current) * 0.5
    stable_count = 0
    accepted_moves = 0

    for iter_count in range(max_iter):
        for i in range(len(x_current)):  # Εξερεύνηση κατά μήκος αξόνων
            x_new = x_current.copy()
            x_new[i] += np.random.uniform(-step_size[i], step_size[i])
            x_new = gradient_descent(func, grad, x_new)  # Gradient Descent
            
            f_current = func(x_current)
            f_new = func(x_new)

            # Metropolis criterion: αποδοχή χειρότερων λύσεων με πιθανότητα
            if f_new < f_current or np.random.rand() < np.exp((f_current - f_new) / temp):
                x_current = x_new
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new
                accepted_moves += 1

            # Προσαρμογή μήκους βήματος κάθε Ns βήματα
            if iter_count % Ns == 0:
                acceptance_ratio = accepted_moves / Ns
                if acceptance_ratio > 0.6:
                    step_size[i] *= (1 + 0.1 * (acceptance_ratio - 0.6))
                elif acceptance_ratio < 0.4:
                    step_size[i] /= (1 + 0.1 * (0.4 - acceptance_ratio))
                accepted_moves = 0

        temp *= alpha  # Μείωση θερμοκρασίας
        if np.linalg.norm(x_current - best_x) < 1e-5:
            stable_count += 1
        else:
            stable_count = 0
        if stable_count > Nt:  # Αν η λύση είναι σταθερή για Nt μειώσεις θερμοκρασίας, σταμάτα
            break

    return best_x, best_f


# -------------------------- Εκτέλεση της μεθόδου στις 5 συναρτήσεις --------------------------

functions = [(rastrigin, rastrigin_grad, [2, 2]), (camel, camel_grad, [0.5, 0.5]),
             (shubert, shubert_grad, [1, 1]), (griewank2, griewank2_grad, [2, 2]),
             (branin, branin_grad, [1, 1])]

for func, grad, x_init in functions:
    best_x, best_f = simulated_annealing(func, grad, x_init)
    print(f"Function: {func.__name__}")
    print("Best solution:", best_x)
    print("Best function value:", best_f, "\n")

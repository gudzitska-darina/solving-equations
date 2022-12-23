import math
import scipy.misc

eps = 1e-6

def f1(x):
    return math.sqrt(abs(math.sin(x))) + math.pow(x, 3) - math.cos(x) - x - 10


def f2(x):
    return 13 * math.sin(x) * (math.log((x * math.exp(x) + 32), 2) + math.sinh(math.pow(x, 2))) - math.pow(x, 2)


def f3(x):
    return -136 * math.pow(x, 7) + 24 * math.pow(x, 6) + 650 * math.pow(x, 5) - \
           124 * math.pow(x, 4) - 795 * math.pow(x, 3) + 145 * math.pow(x, 2) + 157 * x - 1


def fourier_condition(func, x):
    second_derivative = scipy.misc.derivative(func, x, dx=1.0, n=2)
    return func(x) * second_derivative > 0


def simple_iteration(a, b) -> float:
    root = 0.0
    alpha = min(scipy.misc.derivative(f1, a, dx=1.0, n=1), scipy.misc.derivative(f1, b, dx=1.0, n=1))
    gamma = max(scipy.misc.derivative(f1, a, dx=1.0, n=1), scipy.misc.derivative(f1, b, dx=1.0, n=1))
    lambd = 2 / (gamma + alpha)
    q = abs((gamma - alpha) / (gamma + alpha))
    prew = a
    temp = b

    while abs(temp - prew) > (1 - q) / q * eps:
        prew = temp
        temp = prew - lambd * f1(prew)
        root = temp

    return root


def bisection(a, b) -> float:
    root = 0.0
    while True:
        mid = (a + b) / 2
        if mid == 0 or abs(b - a) < abs(eps):
            root = mid
            break

        if f2(a) * f2(mid) < 0:
            b = mid
        elif f2(b) * f2(mid) < 0:
            a = mid

    return root


def simplified_newton(a, b) -> float:
    root = 0.0
    fixed_point = 0.0
    minimum = min(abs(scipy.misc.derivative(f2, a, dx=1.0, n=1)), abs(scipy.misc.derivative(f2, b, dx=1.0, n=1)))

    if fourier_condition(f2, a):
        fixed_point = a
    else:
        fixed_point = b

    root = fixed_point
    derivate_x0 = scipy.misc.derivative(f2, fixed_point, dx=1.0, n=1)
    while True:
        root = root - f2(root) / derivate_x0
        if abs(f2(root)) <= minimum * eps:
            break

    return root


def normalize(arr) -> list:
    res = []
    n = 0
    for i in arr:
        n += math.pow(i, 2)
    for a in arr:
        res.append(a / math.sqrt(n))
    return res


def stop_criteria(old, new) -> bool:
    for o, n in zip(old, new):
        diff = abs(math.pow(o, 2) - n)
        if diff > eps:
            return False
    return True


def lobachevski(coefficients) -> list:
    a = normalize(coefficients)
    b = [1] * len(coefficients)
    n = len(coefficients) - 1
    p = 0
    while not stop_criteria(a, b):
        p += 1
        for k in range(0, n+1):
            sum = 0
            for j in range(1, min(k, n-k)+1):
                sum += math.pow(-1, j) * a[k - j] * a[k + j]
            b[k] = math.pow(a[k], 2) + 2 * sum

        a = normalize(b).copy()

    res = [0] * (len(b) - 1)
    power = math.pow(2, -p)
    for i in range(1, len(b)):
        root = math.pow(b[i] / b[i - 1], power)
        if abs(f3(root)) < 5:
            res[i - 1] = root
        else:
            res[i - 1] = -root

    return res


def check_lobachevski(a, b) -> float:
    root = 0
    while True:
        mid = (a + b) / 2
        if mid == 0 or abs(b - a) < abs(eps):
            root = mid
            break

        if f3(a) * f3(mid) < 0:
            b = mid
        elif f3(b) * f3(mid) < 0:
            a = mid
    return root


def main():
    print("Simple iteration method for f1: ")
    print(f"\t{simple_iteration(1.5, 2.5)}\n")

    print("Bisection method for f2: ")
    print(f"\t{bisection(-3.5, -3)}")
    print(f"\t{bisection(-0.5, 0.5)}")
    print(f"\t{bisection(3, 3.5)}\n")

    print("Simplified Newton`s method for f2: ")
    print(f"\t{simplified_newton(-3.5, -3)}")
    print(f"\t{simplified_newton(-0.5, 0.5)}")
    print(f"\t{simplified_newton(3, 3.5)}\n")

    print("Lobachevski: ")
    roots = lobachevski([-136, 24, 650, -124, -795, 145, 157, -1])
    for _ in roots:
        print(f"\t{_}")

    print("\nCheck Lobachevski (bisection): ")
    print(f"\t{check_lobachevski(-2, -1.5)}")
    print(f"\t{check_lobachevski(-1.5, -1)}")
    print(f"\t{check_lobachevski(-0.5, 0)}")
    print(f"\t{check_lobachevski(0, 0.5)}")
    print(f"\t{check_lobachevski(0.5, 1)}")
    print(f"\t{check_lobachevski(1, 1.5)}")
    print(f"\t{check_lobachevski(1.5, 2)}\n")


if __name__ == '__main__':
    main()


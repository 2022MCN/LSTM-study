import random
from knapsack import *

def testNK01():
    n = 5
    capacity = 20
    weight_cost = [(random.randint(1, 20), random.randint(20, 100)) for e in range(n)]
    print(weight_cost)
    kn = Knapsack()
    print("Brute Force solution")
    best_cost, solution = kn.brute_force(capacity, weight_cost)
    print("Cost", best_cost, ", Solution =", solution) #1 == taken, 0 == not taken

    print("\nDynamic Programming solution")
    T = kn.knapsackDP(capacity, weight_cost)
    for r in T:
        print(r)

    print("\nGreedy solution")
    cUsed, cost, solution = kn.ratio_greedy(capacity, weight_cost)
    print("Capacity Used =", cUsed, ", Cost =", cost, ", Solution =", solution)

    print("\nBranch and Bound solution")
    cost, solution = kn.knBnB(capacity, weight_cost)
    print("Cost =", cost, ", Solution =", solution)

def main():
    testNK01()

if __name__ == '__main__':
    main()
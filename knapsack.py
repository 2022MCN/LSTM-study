from itertools import combinations
from queue import PriorityQueue
from queue import Queue
from queue import LifoQueue

class Node(): #This class is intializing the node
    def __init__(self, level, selected_items, cost, weight, bound):
        self.level = level
        self.selected_items = selected_items
        self.cost = cost
        self.weight = weight
        self.bound = bound
    
    def __lt__(self, other):
        return self.bound < other.bound

    def __repr__(self):
        return str(self.level) + " " + str(self.selected_items) + " " + \
            str(self.cost) + " " + str(self.weight) + " " + str(self.bound)

class Knapsack():
    #This function is solving the Knapsack problem using brute force
    def brute_force(self, capacity, weight_cost):
        N = len(weight_cost)
        best_cost = None
        solution = []
        #This loop is calculating the best cost that can be carried from a given weight
        for way in range(N):
            for comb in combinations(weight_cost, way + 1): #Enter the number of all cases
                weight = sum([wc[0] for wc in comb]) 
                cost = sum([wc[1] for wc in comb]) 
                #Compare best cost with current cost and Check the weight is smaller than capacity
                if(best_cost is None or best_cost < cost) and weight <= capacity:
                    best_cost = cost #Update the best cost
                    solution = [0] * N
                    for wc in comb: 
                        solution[weight_cost.index(wc)] = 1 #Mark the selected weight by 1 in the solution
        return best_cost, solution

    #This function is solving the Knapsack problem using dynamic programming
    def knapsackDP(self, capacity, weight_cost):
        N = len(weight_cost)
        T = [[0 for x in range(capacity + 1)] for y in range(N + 1)] #Make the table for DP

        for i in range(1, N + 1): #Looping through the elements of array
            for j in range(capacity + 1):
                if weight_cost[i-1][0] > j: #If i-1 can't put in knapsack because of weight
                    T[i][j] = T[i-1][j] #Put the maximum value that can be put in i-th bag
                else: #If it can put in knapsack
                    #Using max function to select larger one
                    T[i][j] = max(T[i-1][j], T[i-1][j - weight_cost[i - 1][0]] + weight_cost[i - 1][1])
        return T

    #This function is solving the Knapsack problem using greedy algorithm
    def ratio_greedy(self, capacity, weight_cost):
        N = len(weight_cost)
        #Create ratios(0 is weight, 1 is value)
        ratios = [(index, item[1] / float(item[0])) for index, item in enumerate(weight_cost)]
        ratios = sorted(ratios, key=lambda x: x[1], reverse = True) #Sorting based on values and reverse
        solution = [0] * N 
        cost = 0
        cUsed = 0
        for index, ratio in ratios:
            #If the weight cost of the item is less than the capacity
            if weight_cost[index][0] + cUsed <= capacity:
                cUsed += weight_cost[index][0] #Put items in the knapsack
                cost += weight_cost[index][1]
                solution[index] = 1 #Mark the selected weight by 1 in the solution
            print("Capacity Used =", cUsed, ", Cost", cost, ", Solution =", solution)
        return cUsed, cost, solution

    #This function is solving the Knapsack problem using branch and bound
    def knBnB(self, capacity, weight_cost):
        N = len(weight_cost)
        PQ = PriorityQueue()
        ratios = [(index, item[1] / float(item[0])) for index, item in enumerate(weight_cost)]
        ratios = sorted(ratios, key=lambda x: x[1], reverse = True)
        print(ratios)
        best_so_far = Node(0, [], 0.0, 0.0, 0.0) #Empty node
        rBound = self.calculate_bound(best_so_far, capacity, weight_cost, ratios) #Root node
        root = Node(0, [], 0.0, 0.0, rBound)

        PQ.put(root)
        print(root)

        while not PQ.empty():
            cNode = PQ.get() #Getting the element

            if cNode.bound > best_so_far.cost: #If bound is better than best solution(Check if it is promissing)
                cNode_index = ratios[cNode.level][0] #Get the index of current node
                next_item_cost = weight_cost[cNode_index][1] #Get the next item cost
                next_item_weight = weight_cost[cNode_index][0] #Get the next item weight

                next_added = Node( #Create left child which is taken the item
                    cNode.level + 1,
                    cNode.selected_items + [cNode_index],
                    cNode.cost + next_item_cost, #Add the next item cost
                    cNode.weight + next_item_weight, #Add the next item weight
                    cNode.bound
                )

                if next_added.weight <= capacity:
                    if next_added.cost > best_so_far.cost: #This condition is for updated cost 
                        #Check the updated cost is larger than best solution
                        best_so_far = next_added #Update the best solution
                    #Update to next node's bound
                    next_added.bound = self.calculate_bound(next_added, capacity, weight_cost, ratios)
                    if next_added.bound > best_so_far.cost: #Check the updated bound is better than best solution
                        PQ.put(next_added) #Put node in PQ
                        print(next_added)
                
                next_not_added = Node( #Create right child which is not taken the item
                    cNode.level + 1,
                    cNode.selected_items,
                    cNode.cost, #Keep the previous cost
                    cNode.weight, #Keep the previous weight
                    cNode.bound
                )
                #Update to next node's bound
                next_not_added.bound = self.calculate_bound(next_not_added, capacity, weight_cost, ratios)
                if next_not_added.bound > best_so_far.cost: #Check the current bound is better than best solution
                    PQ.put(next_not_added) #Put node in PQ
                    print(next_not_added)
        
        solution = [0] * N

        for wc in best_so_far.selected_items: #Mark the selected items by 1 in the solution
            solution[wc] = 1
        return int(best_so_far.cost), solution


    #This function is calculating bound
    def calculate_bound(self, node, capacity, weight_cost, ratios):
        N = len(weight_cost)

        if node.weight >= capacity: #If weight is larger than capacity(=very smaller bound)
            return 0
        else: #If weight is smaller than capacity
            bound = node.cost
            total_weight = node.weight
            current_level = node.level

        while current_level < N: #This loop is running for all remaining elements(i-th element < total element)
            current_index = ratios[current_level][0]
            if total_weight + weight_cost[current_index][0] > capacity: #If best solution is larger than capacity(k-1)
                cost = weight_cost[current_index][1]
                weight = weight_cost[current_index][0]
                #Calculate the max profit that can get from current condition
                bound += (capacity - total_weight) * cost / weight
                break
            #If best solution is smaller than capacity(k)
            bound += weight_cost[current_index][1]
            total_weight += weight_cost[current_index][0]
            current_level += 1

        return bound
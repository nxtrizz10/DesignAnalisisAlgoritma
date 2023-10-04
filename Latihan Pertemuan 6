# Dynamic Programming deret Fibonacci

# FIRST TOW TERMS

nterms = int(input("How many terms? "))
n1, n2 = 0, 1
count = 0

if nterms <= 0:
  print("Please enter a positive integer")
elif nterms == 1:
  print("Fibonacci sequence upto", nterms, ":")
  print(n1)
else:
  print("Fibonacci sequence: ")
  while count < nterms:
    print(n1)
    nth = n1 + n2
    # Update values
    n1 = n2
    n2 = nth
    count += 1

# Fibonacci dengan rekursi

def recur_fibo(n):
  if n <= 1:
    return n
  else:
    return(recur_fibo(n-1) + recur_fibo(n-2))

# Memakai input
nterms = int(input("How many terms? "))

# Cek nilai nterms
if nterms <= 0:
  print("Please enter a positive integer")
else:
  print("Fibonacci sequence: ")
  for i in range(nterms):
    print(recur_fibo(i))



# The Travelling Salesperson Problem
import matplotlib
import matplotlib.pyplot as plt
import random
import time
import itertools

# Try all tours (exact_TSP)
def exact_TSP(cities):
  "Generate all possible tours of the cities and choose te shortest one."
  return shortest(alltours(cities))

def shortest(tours):
  "Return the tour with the minimum total distance."
  return min(tours, key = total_distance)

# Representing Tours
alltours = itertools.permutations

cities = {1, 2, 3}

list(alltours(cities))

# Representing Cities and Distance
def total_distance(tour):
    "The total distance between each pair of consecutive cities in the tour."
    return sum(distance(tour[i], tour[i-1]) for i in range(len(tour)))

City = complex

def distance(A, B):
  "The distance between two points."
  return abs(A - B)

A = City(300, 0)
B = City(0, 400)

distance(A, B)

# Make a set of random Cities
def Cities(n):
  "Make a set of n cities, each with random coordinates."
  return set(City(random.randrange(10, 890), random.randrange(10, 590)) for c in range(n))

# Let's make some standard sets of cities of various sizes
# We'll set the random seed so that these sets are the same every time we run this notebook
random.seed('seed')
cities8, cities10, cities100, cities1000, cities10000= Cities(8), Cities(10), Cities(100), Cities(1000), Cities(10000)
cities8

# Apply the exact TSP function to find the minimal tour

tour = exact_TSP(cities8)

print(tour)
print(total_distance(tour))

# Try All Non-Redundant Tours
def alltours(cities):
  "Return a list of tours, each a permutation of cities, but each one starting with the same city."
  start = first(cities)
  return [[start] + list(tour)
  for tour in itertools.permutations(cities - {start})]

def first(collection):
  "Start iterating over collection, and return the first element."
  for x in collection:
    return x

# We can verify that for 3 cities there are now 2 tours (not 6) and for 4 cities there are 6 tours (not 24).

alltours({1, 2, 3})

alltours({1, 2, 3, 4})

# We can also verify that calling exact_TSP(cities8) still works and gives the same tour with the same total distance.
# (But it is now about 8 times faster.)
tour = exact_TSP(cities8)

print(tour)
print(total_distance(tour))

# Plotting
def plot_tour(algorithm, cities):
  "Apply a TSP algorithm to cities, and plot the resulting tour."
  # Find the solution and time long it takes
  t0 = time.time()
  tour = algorithm(cities)
  t1 = time.time()
  # Plot the tour as blue lines between blue circles, and the starting city as a red square.
  plotline(list(tour) + [tour[0]])
  plotline([tour[0]], 'rs')
  plt.show
  print("{} city tour; total distance = {:.1f}; time = {:.3f} secs for {}".format(
      len(tour), total_distance(tour), t1-t0, algorithm.__name__))

def plotline(points, style = 'bo-'):
  "Plot a list of points (complex numbers) in the 2-D plane."
  X, Y = XY(points)
  plt.plot(X, Y, style)

def XY(points):
  "Given a list of points, return two lists: X coordinates, and Y coordinates."
  return [p.real for p in points], [p.imag for p in points]

plot_tour(exact_TSP, cities8)

plot_tour(exact_TSP, cities10)

# Greedy Nearest Neighbor (greedy_TSP)
def greedy_TSP(cities):
  "At each step, visit the nearest neighbor that is still unvisited"
  start = first(cities)
  tour = [start]
  unvisited = cities - {start}
  while unvisited:
    C = nearest_neighbor(tour[-1], unvisited)
    tour.append(C)
    unvisited.remove(C)
  return tour

def nearest_neighbor(A, cities):
  "Find the city in cities that is nearest to city A."
  return min(cities, key = lambda x: distance(x, A))

cities = Cities(9)
plot_tour(exact_TSP, cities)

plot_tour(greedy_TSP, cities)

plot_tour(greedy_TSP, cities100)

plot_tour(greedy_TSP, cities1000)

# Algorithm 3: Greed Neares Neighbor from All starting Points (all_greedy_TSP)

def all_greedy_TSP(cities):
  "Try the greedy algorithm from each of the starting cities: return the shortest tour."
  return shortest(greedy_TSP(cities, start=c) for c in cities)

# Modify greedy_TSP to take an optimal start city: otherwise it is unchanged.

def greedy_TSP(cities, start = None):
  "At each step, visit the nearest neighbor that is still unvisited."
  if start is None:
    start = first(cities)
    tour = [start]
    unvisited = cities - {start}
    while unvisited:
      C = nearest_neighbor(tour[-1], unvisited)
      tour.append(C)
      unvisited.remove(C)
    return tour

# Compare greedy_TSP to all_greedy_TSP
plot_tour(greedy_TSP, cities100)

plot_tour(all_greedy_TSP, cities100)

# Algorithm 4: Greedy Nearest Neighbor with Exact End (greedy_exact_end_TSP)

def greedy_exact_end_TSP(cities, start=None, end_size=8):
  "At each step, visit the nearest neighbor that is still unvisited untill there are k_end cities left; then choose the best ofall possible endings."
  if start is None:
    start = first(cities)
    tour = [start]
    unvisited = cities - {start}
    # Use greedy algoritm for all but the last end_size cities
    while len(unvisited) > end_size:
      C = nearest_neighbor(tour[-1], unvisited)
      tour.append(C)
      unvisited.remove(C)

    ends = map(list, itertools.permutations(unvisited))
    best = shortest([tour[0], tour[-1]] + end for end in ends)
    return tour + best[2:]

plot_tour(greedy_exact_end_TSP, cities100)

plot_tour(greedy_exact_end_TSP, cities1000)

# Algorithm 5: Greedy Nearest Neighbor with Both Ends Search (greedy_bi_TSP)

def greedy_bi_TSP(cities, start_size=12, end_size=6):
  "At each step, visit the nearest neighbor that is still unvisited."
  starts = random.sample(cities, min(len(cities), start_size))
  return shortest(greedy_exact_end_TSP(cities, start, end_size) for start in starts)

random.seed('bi')
plot_tour(greedy_bi_TSP, cities100)
plot_tour(greedy_bi_TSP, cities1000)

# Benchmarking Algorithms
def compare_algorithms(algorithms, maps):
  "Apply each algorithm to each map and plot results."
  for algorithm in algorithms:
    t0 = time.time()
    results = [total_distance(algorithm(m)) for m in maps]
    t1 = time.time()
    avg = sum(results) / len(results)
    label = '{:.0f}; {:.1f}s: {}'.format(avg, t1-t0, algorithm.__name__)
    plt.plot(sorted(results), label=label)
  plt.legend(loc=2)
  plt.show()
  print('{} x {}-city maps'.format(len(maps), len(maps[0])))

def Maps(M, N):
  "Return a list of M maps, each consisting of a set of N cities."
  return [Cities(N) for m in range(M)]

compare_algorithms([greedy_TSP, greedy_exact_end_TSP, all_greedy_TSP], Maps(100, 50))

# Algoritma Geedy: Huffman Coding
string = 'BCAADDDCCACACAC'

#Creating tree nodes
class NodeTree(object):
  def __init__(self, left = None, right = None):
    self.left = left
    self.right = right

  def children(self):
    return (self.left, self.right)

  def nodes(self):
    return (self.left, self.right)

  def __str__(self):
    return '%s_%s' % (self.left, self.right)

# Main Function implemening huffman coding
def huffman_code_tree(node, left = True, binString = ''):
  if type(node) is str:
    return {node: binString}
  (l, r) = node.children()
  d = dict()
  d.update(huffman_code_tree(l, True, binString + '0'))
  d.update(huffman_code_tree(r, False, binString + '1'))
  return d

# Calculating frequency
freq = {}
for c in string:
  if c in freq:
    freq[c] += 1
  else:
    freq[c] = 1

freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

nodes = freq

while len(nodes) > 1:
  (key1, c1) = nodes[-1]
  (key2, c2) = nodes[-2]
  nodes = nodes[:-2]
  node = NodeTree(key1, key2)
  nodes.append((node, c1 + c2))

  nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

huffmanCode = huffman_code_tree(nodes[0][0])

print(' Char | Huffman code ')
print('----------------------')
for (char, frequency) in freq:
  print(' %-4r | %12s' % (char, huffmanCode[char]))

# A Huffman Tree Node
import heapq

class node:
  def __init__(self, freq, symbol, left=None, right=None):
     # frequency of symbol
     self.freq = freq

     # symbol name (character)
     self.symbol = symbol

     # node left of current node
     self.left = left

     # node right of current node
     self.right = right

     # tree direction (0/1)
     self.huff = ''
  def __lt__(self, nxt):
    return self.freq < nxt.freq

# utility function to print huffman
# codes for all symbols in the newly
# created Huffman tree
def printNodes(node, val=''):

  # huffman code for current node
  newVal = val + str(node.huff)

  # if node is not an edge node
  # then traverse inside it
  if(node.left):
    printNodes(node.left, newVal)
  if(node.left):
    printNodes(node.right, newVal)

    # if node is edge node then
    # display its huffman code
  if(not node.left and not node.right):
    print(f"{node.symbol} -> {newVal}")

# characters for huffman tree
chars = ['a', 'b', 'c', 'd', 'e', 'f']

# frequency of characters
freq = [5, 9, 12, 13, 16, 45]

# list containing unused nodes
nodes = []

# convertiing characters and frequencies
# into huffman tree nodes
for x in range(len(chars)):
  heapq.heappush(nodes, node(freq[x], chars[x]))

while len(nodes) > 1:

  # sort all the nodes in ascending order
  # based on their frequency
  left = heapq.heappop(nodes)
  right = heapq.heappop(nodes)

  # assign directional value to these nodes
  left.huff = 0
  right.huff = 1

  # combine the 2 smallest nodes to create
  # new node as their parrent
  newNode = node(left.freq + right.freq, left.symbol + right.symbol, left, right)

  heapq.heappush(nodes, newNode)

# Huffman Tree is readyy
printNodes(nodes[0])

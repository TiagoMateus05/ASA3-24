from pulp import *

def read_input():
    numFactories, numCountries, numKids = map(int, input().split())

    factories = []
    for _ in range(numFactories):
        factories.append(list(map(int, input().split())))

    countries = []
    for _ in range(numCountries):
        countries.append(list(map(int, input().split())))

    kids = []
    for _ in range(numKids):
        kids.append(list(map(int, input().split())))

    return factories, countries, kids

factories, countries, kids = read_input()

prob = LpProblem("P3", LpMaximize)

x = {}

for factory in factories:
    i = factory[0]
    for k in range(1, len(kids) + 1):
        x[(i, k)] = LpVariable(f"x_{i}_{k}", cat="Binary")

# Objective function: Maximize the number of satisfied kids
prob += sum(x[(i, k)] for i, k in x.keys()), "Maximize_Satisfied_Kids"

# Export limits per country
for country in countries:
    j, export_max, _ = country
    factories_in_country = [f[0] for f in factories if f[1] == j]
    prob += (
        sum(x[(i, k)] for i in factories_in_country for k in range(1, len(kids) + 1)) <= export_max,
        f"Export_Limit_Country_{j}"
    )

# Minimum toy distribution per country
for country in countries:
    j, _, min_toys = country
    factories_in_country = [f[0] for f in factories if f[1] == j]
    prob += (
        sum(x[(i, k)] for i in factories_in_country for k in range(1, len(kids) + 1)) >= min_toys,
        f"Min_Distribution_Country_{j}"
    )

# Each kid can receive at most one toy
for kid in kids:
    k = kid[0]
    requested_factories = kid[2:]  # List of factory IDs requested by this kid
    prob += (
        sum(x[(i, k)] for i in requested_factories) <= 1,
        f"Kid_Constraint_{k}"
    )

# Each factory can satisfy at most its maximum stock
for factory in factories:
    i, _, max_stock = factory
    prob += (
        sum(x[(i, k)] for k in range(1, len(kids) + 1)) <= max_stock,
        f"Factory_Stock_Constraint_{i}"
    )

# Ensure each kid can only receive toys from requested factories
for kid in kids:
    k = kid[0]
    requested_factories = kid[2:]
    for factory in factories:
        i = factory[0]
        if i not in requested_factories:
            prob += x[(i, k)] == 0, f"Kid_Factory_Constraint_{k}_{i}"

prob.solve(GLPK(msg=0))

if prob.status == 1:  # Check if the problem is solved
    print(prob.objective.value())
else:
    print(-1)
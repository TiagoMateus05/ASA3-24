from pulp import *

def read_input():
    try:
        # Read the first line: number of factories, countries, and kids
        numFactories, numCountries, numKids = map(int, input().split())
        if numFactories < 1 or numCountries < 1 or numKids < 1:
            raise ValueError("The number of factories, countries, and kids must be greater than 0.")

        factories = []
        for _ in range(numFactories):
            factory = list(map(int, input().split()))
            if len(factory) != 3:
                raise ValueError("Each factory must have 3 elements: ID, country, and stock.")
            factories.append(factory)

        countries = []
        for _ in range(numCountries):
            country = list(map(int, input().split()))
            if len(country) != 3:
                raise ValueError("Each country must have 3 elements: ID, export_max, and min_toys.")
            countries.append(country)

        kids = []
        for _ in range(numKids):
            kid = list(map(int, input().split()))
            if len(kid) < 3:
                raise ValueError("Each kid must have at least 3 elements: ID, country, and at least one requested factory.")
            kids.append(kid)

        return factories, countries, kids
    except ValueError as e:
        print(f"Error: {e}")
        return None, None, None

# Read input
factories, countries, kids = read_input()
if not factories or not countries or not kids:
    print(-1)  # Exit if input is invalid or empty
    exit()

# Create the optimization problem
prob = LpProblem("P3", LpMaximize)

# Precompute valid pairs and country factories
valid_pairs = {
    (i, k)
    for kid in kids
    for k, kid_country, *requested_factories in [kid]  # Desestruturação da criança
    for i in requested_factories  # Fábricas solicitadas pela criança
}

country_factories = {j: [f[0] for f in factories if f[1] == j] for j in range(1, len(countries) + 1)}
kid_countries = {kid[0]: kid[1] for kid in kids}

# Define decision variables only for valid pairs where the factory has stock > 0
x = {
    (i, k): LpVariable(f"x_{i}_{k}", cat="Binary")
    for i, _, stock in factories if stock > 0  # Fábricas com estoque > 0
    for k, kid_country, *requested_factories in kids  # Para cada criança
    for requested_factory in requested_factories  # Para cada fábrica solicitada pela criança
    if i == requested_factory  # Apenas fábricas solicitadas pela criança
}

# Objective function: Maximize the number of satisfied kids
prob += lpSum(x.values()), "Maximize_Satisfied_Kids"

# Export limits and minimum toy distribution for countries
for j, export_max, min_toys in countries:
    factories_in_country = country_factories.get(j, [])

    # If there are no factories in a country, skip to the next country
    if not factories_in_country:
        continue

    # Export constraint: Only count toys sent to kids in other countries (not the same country)
    prob += (
        lpSum(
            x[(i, k)]
            for i in factories_in_country
            for k in range(1, len(kids) + 1)
            if (i, k) in x and kid_countries[k] != j  # Exclude kids in the same country
        ) <= export_max,
        f"Export_Limit_Country_{j}"
    )

    # Minimum toy distribution: Includes all kids in this country (same country as factory)
    prob += (
        lpSum(
            x[(i, k)]
            for i in factories_in_country
            for k in range(1, len(kids) + 1)
            if (i, k) in x
        ) >= min_toys,
        f"Min_Distribution_Country_{j}"
    )

# Each kid can receive at most one toy (only from requested factories)
for kid in kids:
    k = kid[0]
    requested_factories = kid[2:]  # Factories requested by the kid
    prob += (
        lpSum(x[(i, k)] for i in requested_factories if (i, k) in x) <= 1,
        f"Kid_Constraint_{k}"
    )

# Each factory can satisfy at most its maximum stock
for factory in factories:
    i, _, max_stock = factory
    prob += (
        lpSum(x[(i, k)] for k in range(1, len(kids) + 1) if (i, k) in x) <= max_stock,
        f"Factory_Stock_Constraint_{i}"
    )

# Solve the problem using GLPK solver
prob.solve(GLPK(msg=0))

# Output the result
if prob.status == 1:  # Check if the problem is solved
    print(int(prob.objective.value()))
else:
    print(-1)

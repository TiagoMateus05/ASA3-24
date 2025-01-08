#!/usr/bin/env python3
import sys
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, LpBinary
from pulp import GLPK_CMD

def main():
    line = sys.stdin.readline().strip()
    if not line:
        print(-1)
        return
    n, m, t = map(int, line.split())

    factories = {}
    # For each factory i, how many (k,i) pairs exist
    factory_potential_usage = [0] * (n + 1)

    countries = {}
    country_num_children = [0] * (m + 1)
    # We'll build adjacency to quickly handle constraints:
    #  - factory_children[i] will hold all children that requested factory i
    #  - country_children[j] will hold all children that live in country j
    #  - export_pairs[j] will hold all (child, factory) pairs where factory is in j but child is in a different country
    factory_children = [[] for _ in range(n + 1)]
    country_children = [[] for _ in range(m + 1)]
    export_pairs = [[] for _ in range(m + 1)]

    # Will store for each child k -> (country_of_child, [factories requested])
    child_requests = {}
    # Decision variables: x[(k,i)] = 1 if child k gets the toy from factory i
    x = {}

    # Read factory data
    for _ in range(n):
        line = sys.stdin.readline().strip()
        if not line:
            print(-1)
            return
        i_factory, j_country, f_max = map(int, line.split())
        factories[i_factory] = (j_country, f_max)

    # Read countries
    for _ in range(m):
        line = sys.stdin.readline().strip()
        if not line:
            print(-1)
            return
        j_country, pmax_j, pmin_j = map(int, line.split())
        countries[j_country] = (pmax_j, pmin_j)

    # Create the LP problem
    prob = LpProblem(sense=LpMaximize)

    # Read children and immediately create x variables
    for _ in range(t):
        line = sys.stdin.readline().strip()
        if not line:
            print(-1)
            return
        parts = list(map(int, line.split()))
        k_child = parts[0]
        j_child_country = parts[1]
        wanted_factories = parts[2:]
        country_num_children[j_child_country] += 1
        child_requests[k_child] = (j_child_country, wanted_factories)

        for i_fact in wanted_factories:
            if i_fact not in factories:
                print(-1)
                return
            var = LpVariable(cat=LpBinary, name=f"x_{k_child}_{i_fact}")
            x[(k_child, i_fact)] = var
            factory_potential_usage[i_fact] += 1

    # Build adjacency lists now that we know x
    for (k_child, i_fact), var in x.items():
        # child k_child wants factory i_fact
        j_fact_country, _ = factories[i_fact]
        j_child_country = child_requests[k_child][0]

        factory_children[i_fact].append(k_child)

        if j_fact_country != j_child_country:
            export_pairs[j_fact_country].append((k_child, i_fact))

    # Also gather which children are in which country
    for k_child, (j_country_child, _) in child_requests.items():
        country_children[j_country_child].append(k_child)

    # Objective: sum of all x[(k,i)]
    prob += lpSum(x.values())

    # 1) Each child can get at most 1 present
    for k_child, (j_c, facs) in child_requests.items():
        prob += lpSum(x[(k_child, i)] for i in facs) <= 1

    # 2) Factory capacity
    for i_fact in range(1, n + 1):
        if i_fact not in factories:
            continue
        j_fact_country, f_max = factories[i_fact]
        pot_usage = factory_potential_usage[i_fact]
        if pot_usage > 0 and f_max < pot_usage:
            # sum of x[(k, i_fact)] over all children k that requested i_fact
            prob += lpSum(x[(k, i_fact)] for k in factory_children[i_fact]) <= f_max

    # 3) pmin_j: minimum number of presents to children in each country j
    for j_country in range(1, m + 1):
        if j_country not in countries:
            continue
        pmax_j, pmin_j = countries[j_country]
        if pmin_j > 0:
            if pmin_j > len(country_children[j_country]):
                print(-1)
                return
            # sum_{k in that country} sum_{i in the factories requested by k} x[(k,i)] >= pmin_j
            # But we can do a direct sum:
            c_kids = country_children[j_country]
            prob += lpSum(
                lpSum(x[(k, i)] for i in child_requests[k][1])
                for k in c_kids
            ) >= pmin_j

    # 4) pmax_j: max exports from factories of country j to children in different countries
    # First, figure out the potential exports from j:
    for j_country in range(1, m + 1):
        if j_country not in countries:
            continue
        pmax_j, pmin_j = countries[j_country]
        pairs = export_pairs[j_country]
        if len(pairs) > pmax_j:
            # sum_{(k, i_fact) in pairs} x[(k,i_fact)] <= pmax_j
            prob += lpSum(x[(k, i_fact)] for (k, i_fact) in pairs) <= pmax_j

    # Solve with GLPK
    prob.solve(GLPK_CMD(msg=0))

    if LpStatus[prob.status] not in ["Optimal", "Feasible"]:
        print(-1)
        return

    print(int(round(prob.objective.value())))

if __name__ == "__main__":
    main()

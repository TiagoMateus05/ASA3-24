import random
import sys

def generate_random_integer(c, x):
    # Generate a random number from a Gaussian (normal) didstribution
    return int(round(random.gauss(c, x)))
    
def generate_request(requests, facts): 
	r = int(random.uniform(1, facts))
	if requests.get(r) is None:
		requests[r] = True 
		return r
	else: 
		return generate_request(requests, facts)
		
def print_usage(argv, msg=None):
    if msg: print('Error:',msg)
    print(f"Usage: python3 {argv[0]} <num_factories> <num_countries> <num_children> <max_factory_capacity>")
    exit(1)
    
# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print_usage(sys.argv, 'Missing arguments')

    num_factories = int(sys.argv[1])
    num_countries = int(sys.argv[2])
    num_children  = int(sys.argv[3])
    variance  = float(sys.argv[4])
    max_cap = int(sys.argv[5])
    if max_cap <= 0: 
        print_usage(sys.argv, 'max_factory_capacity must be greater than 0')
    max_requests = int(sys.argv[6])
    if max_requests <= 0:
        print_usage(sys.argv, 'max_requests must be greater than 0')

    fs_per_country = []
    cs_per_country = []

    avg_fs_per_country = int(num_factories/num_countries)
    avg_cs_per_country = int(num_children/num_countries)

    total_fs = 0
    total_cs = 0

    factories_data = {}
    countries_data = {}
    children_data  = {}

    countries_info = []
    cur_fact = 1
    cur_child = 1


    for c in range(num_countries):
        
        cur_fs = generate_random_integer(avg_fs_per_country, avg_fs_per_country*variance)
        cur_cs = generate_random_integer(avg_cs_per_country, avg_cs_per_country*variance)
        
        if (num_factories - total_fs < cur_fs or c == (num_countries-1)):
            cur_fs = num_factories - total_fs
        
        if (num_children - total_cs < cur_cs or c == (num_countries-1)):
            cur_cs = num_children - total_cs
        
        countries_info.append((cur_fs, cur_cs))
        total_fs += cur_fs
        total_cs += cur_cs
        
        cur_total_cap = 0
        for j in range(cur_fact, total_fs+1):
            #print(f"cur fact: {j} in country {c}")
            cap = int(random.uniform(1, max_cap))
            cur_total_cap += cap
            factories_data[j] = (j, c+1, cap)
        
        country_export_cap = int(random.uniform(cur_total_cap/4, cur_total_cap))
        country_min_cs = int(random.uniform(cur_cs/4, cur_cs))
        countries_data[c+1] = (c+1, country_export_cap, country_min_cs)
        
        for ch in range(cur_child, total_cs+1):
            requests_num = int(random.uniform(1, max_requests))
            assert(requests_num < num_factories)
            lst = [ch, c+1]
            requests = {}
            for i in range(requests_num):
                r = generate_request(requests, num_factories)
                lst.append(r)
            children_data[ch] = lst
        
        cur_fact = total_fs+1
        cur_child = total_cs+1
        
    # Print file data
    print(f"{num_factories} {num_countries} {num_children}")
    for i in range(num_factories):
        fi, pj, fmaxi = factories_data[i+1]
        print(f"{fi} {pj} {fmaxi}")
    for i in range(num_countries):
        pj, pmaxj, pminj = countries_data[i+1]
        print(f"{pj} {pmaxj} {pminj}")
    for i in range(num_children):
        c_data = children_data[i+1]
        print(" ".join(map(str, c_data)))


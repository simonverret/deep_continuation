import json
scale_dict = {
        'N': False,
        'R': True
    }

beta_dict = {
        'T10': [10.0],
        'T20': [20.0],
        'T30': [30.0],
        'l3T': [15.0, 20.0, 25.0], 
        'l5T': [10.0, 15.0, 20.0, 25.0, 30.0],
    }

path_dict = {
        'F': 'data/Fournier/valid/',
        'G': 'data/G1/valid/',
        'B': 'data/B1/valid/',
    }

for p, path in path_dict.items():
    for b, beta, in beta_dict.items():
        for s, scale in scale_dict.items():
            paradict = {
                'data':p,
                'scale':scale,
                'beta':beta,
            }
            filename = f"{p+b+s}.json"
            with open(filename, 'w') as fp:
                json.dump(paradict, fp)
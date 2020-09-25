datasets_dict = {
    "Yoon": {
        "seed": [12345, 123],
        "generate": [106000, 10000],
        "path": "data/Fournier/",
        "peak_type": "Gaussian",
        "wmax": 15.0,
        "nmbrs": [[1, 8]],
        "cntrs": [[0.00, 10.0]],
        "wdths": [[0.20, 1.00]],
        "wghts": [[0.20, 1.00]],
        "Legendre": 64
    },
    "Fournier": {
        "seed": [12345, 123],
        "generate": [10000, 500],
        "path": "data/Fournier/",
        "peak_type": "Gaussian",
        "wmax": 15.0,
        "nmbrs": [[1, 1],[1, 21]],
        "cntrs": [[0.00, 0.50], [0.00, 6.00]],
        "wdths": [[0.10, 1.00], [0.10, 1.00]],
        "wghts": [[1.00, 1.00], [1.00, 1.00]],
        "Legendre": 64
    },
    "Xie": {
        "seed": [12345, 123],
        "generate": [10000, 500],
        "path": "data/Fournier/",
        "peak_type": "Gaussian",
        "wmax": 10.0,
        "nmbrs": [[1, 1],[1, 21]],
        "cntrs": [[0.00, 0.50], [0.00, 5.00]],
        "wdths": [[0.10, 1.00], [0.10, 1.00]],
        "wghts": [[1.00, 1.00], [1.00, 1.00]],
        "Legendre": 64
    },
    "B1": {
        "seed": [12345, 123],
        "generate": [50000, 10000],
        "path": "data/G1/",
        "peak_type": "Gaussian",
        "wmax": 20.0,
        "nmbrs": [[0, 4],[0, 6]],
        "cntrs": [[0.00, 0.00], [4.00, 16.0]],
        "wdths": [[0.04, 0.40], [0.04, 0.40]],
        "wghts": [[0.00, 1.00], [0.00, 1.00]],
        "arngs": [[2.00, 5.00], [0.50, 5.00]],
        "brngs": [[2.00, 5.00], [0.50, 5.00]]
    },
    "G1": {
        "seed": [11111,111],
        "generate": [50000, 10000],
        "path": "data/G1/",
        "peak_type": "Gaussian",
        "wmax": 20.0,
        "nmbrs": [[0, 4],[0, 6]],
        "cntrs": [[0.00, 0.00], [4.00, 16.0]],
        "wdths": [[0.04, 0.40], [0.04, 0.40]],
        "wghts": [[0.00, 1.00], [0.00, 1.00]]
    }
}
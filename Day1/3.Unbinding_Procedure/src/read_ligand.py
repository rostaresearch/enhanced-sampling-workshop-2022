def readLigandClusters(filepath):
    clusters = []
    with open(filepath, 'r') as f:
        for line in f:
            clusters.append(set())
            for type in line.split():
                clusters[-1].add(type)
    return clusters

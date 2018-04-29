from pyspark import SparkContext
from pyspark.mllib.random import RandomRDDs
import random
import sys
from kmeans import kmeans
from pyspark import SparkConf
import time

def mean_cluster(k):
    """
    :param k: Nombre de clusters
    :return: Dictionnaire avec moyenne pour chaque cluster
    """
    mean = {}
    for cluster in range(0, k):
        mean[cluster] = random.randint(0, 100)
    print(mean)
    return mean


def generator_normal_rdd(sc, n, d, s, k):
    """
    :param n: Nombre de lignes (de données)
    :param d: Dimension des données (lignes)
    :param s: Ecart type
    :param k: Clusters
    :return: Liste contenant les rdds de points associés à chaque cluster
    """
    normal_rdds = []
    print("LISTE DONNEES : \n")
    for cluster, mean in mean_cluster(k).items():
        normal_rdd = RandomRDDs.logNormalVectorRDD(sc=sc, mean=mean, std=s, numRows=int(n/k), numCols=d, seed=1).map(lambda x: (list(x), cluster))
        print(normal_rdd.collect())
        normal_rdds.append(normal_rdd)
    print()
    return normal_rdds

def export_csv(path, liste):
    with open(path, 'w') as csvfile:
        for rdd in liste:
            for sample in rdd.collect():
                sample_float = [format(i, 'f') for i in sample[0]]
                sample_float.append(sample[1])
                csvfile.write(str(sample_float).replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("'",""))
                csvfile.write('\n')
        csvfile.close()

def export_assign_data(path, rdd):
    with open(path, 'w') as csvfile:
        csvfile.write('Point, Centroid')
        csvfile.write('\n')
        for i in rdd.collect():
            csvfile.write(str(i[0]) + ", " + str(i[1][0]))
            csvfile.write('\n')
        csvfile.close()


if __name__ == "__main__":

    start_time = time.time()

    if len(sys.argv) != 6:
        print("Usage: generator.py out.csv n k d s")
        exit(-1)


    # Create Spark conf
    conf = SparkConf().setAppName("generator").setMaster("local")
    sc = SparkContext(conf=conf)

    path = sys.argv[1]
    n = int(sys.argv[2])
    k = int(sys.argv[3])
    d = int(sys.argv[4])
    s = int(sys.argv[5])

    normalRDD = generator_normal_rdd(sc, n, d, s, k)

    export_csv(path, normalRDD)

    solution = kmeans(sc, path, k, max_iterations=100)
    print("\n\n-----------------------------------------------------------------------------------------------------")
    print("Nombre d'iterations : " + str(solution[2]))
    print("Distance intra cluster : " + str(solution[0]))
    print("Centroids : ")
    for i in solution[1]:
        print("Centroid : " + str(i[0]) + "\t" + str(i[1]))
    print("-----------------------------------------------------------------------------------------------------\n\n")

    export_assign_data("mapped_data.csv", solution[3])

    print("---- %s seconds ----" % (time.time() - start_time))

    sc.stop()
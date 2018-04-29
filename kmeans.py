import math
import time
from pyspark import SparkConf
from pyspark import SparkContext
import sys
import random
dimension = 0

def export_assign_data(path, rdd):
    with open(path, 'w') as csvfile:
        csvfile.write('Point, Centroid')
        csvfile.write('\n')
        for i in rdd.collect():
            csvfile.write(str(i[0]) + ", " + str(i[1][0]))
            csvfile.write('\n')
        csvfile.close()

def avg(x, y):
    return x / y

def convert_float(l, classe):
    l_return = []
    if(classe == True):
        for element in l[:-1]:
            l_return.append(float(element))
        l_return.append(l[-1])
    elif(classe == False):
        for element in l:
            if element == "":
                element = 0
            l_return.append(float(element))
    return l_return

def distance_euclidienne(list_a, list_b):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(list_a, list_b)]))

def moyenne_centroid(m):
    sum_col = [sum([ligne[i] for ligne in m]) for i in range(0, len(m[0]))]
    moyenne = [i/len(m) for i in sum_col]
    return moyenne

def loadData(sc, path):
    rdd_import = sc.textFile(path)
    global dimension
    dimension = len(rdd_import.map(lambda x: x.split(',')).first())

    data = rdd_import\
                        .map(lambda x: (x.split(','), len(x.split(','))))\
                        .filter(lambda x: x[1] == dimension)\
                        .zipWithIndex()\
                        .map(lambda x: (int(x[1]), convert_float(x[0][0], False)))
    return data


def initCentroid_randomRdd(sc, k, d):
    """
    :param sc: SparkContext
    :param k:  Nombre cluster
    :param d: Dimension données
    :return: (id_centroid, [coordonnées points centroid])
    """
    centroids_random = random.RandomRDDs.uniformVectorRDD(sc, k, d - 1).\
                        zipWithIndex().\
                        map(lambda x: (x[1], x[0].tolist()))
    return centroids_random

def initCentroid(sc, k, data):
    """
    :param sc: SparkContext
    :param k:  Nombre cluster
    :param d: Dimension données
    :return: (id_centroid, [coordonnées points centroid])
    """
    centroids_random = sc.parallelize(data.takeSample(False, k)).zipWithIndex().map(
        lambda l: (l[1], l[0][1][:-1]))
    return centroids_random

def assignToCluster(rdd_data, rdd_centroid):
    """
    format (id_point, (id_centroid, distance_euc))
    :return:
    """
    rdd_cartesian = rdd_centroid.cartesian(rdd_data)
    rdd_reduce = rdd_cartesian.map(lambda x: (x[1][0], (x[0][0], distance_euclidienne(x[0][1][:-1], x[1][1][:-1]))))\
                        .groupByKey().mapValues(list)
    point_centroid = rdd_reduce.map(lambda x: (x[0], min(x[1], key=lambda t: t[1])))
    return point_centroid

def computeCentroid(data, point_centroids):
    """"
    retrouver les coordonnées des points --> format (id_centroid, [[coordonnées point 1], [coordonnées point2], [...]])
    :return:
    """
    rdd_map_point_centroid = point_centroids.map(lambda x: (x[0], x[1][0]))
    rdd_join = rdd_map_point_centroid.join(data.map(lambda x: (x[0], x[1][:-1])))
    rdd_join = rdd_join.map(lambda x: (x[1][0], x[1][1]))
    rdd_grouped = rdd_join.groupByKey().mapValues(list)
    rdd_computed_centroid = rdd_grouped.map(lambda x: (x[0], moyenne_centroid(x[1])))
    return rdd_computed_centroid


def computeIntraClusterDistance(distance_point_cluster):
    points_dist_count = distance_point_cluster.map(lambda l: l[1]).map(lambda l: (l[0], 1)).reduceByKey(lambda x, y: x + y)
    points_distance = distance_point_cluster.map(lambda l: l[1]).reduceByKey(lambda x, y: x + y)
    moyenne_distance = points_distance.join(points_dist_count).map(lambda l: (l[0], avg(l[1][0], l[1][1])))
    intra_cluster_distance = moyenne_distance.map(lambda l: l[1]).sum()
    return intra_cluster_distance


def kmeans(sc, path, k, max_iterations, converge=False, moved = 500):
    iteration = 0
    data = loadData(sc, path)
    """
    DECOMMENTE LA LIGNE DU DESSOUS POUR LA QUESTION 10)
    """
    # centroids = initCentroid_randomRdd(sc, k, dimension)
    centroids = initCentroid(sc, k, data)
    print("\n\n-----------------------------------------------------------------------------------------------------")
    print(centroids.collect())
    print("-----------------------------------------------------------------------------------------------------\n\n")

    liste_centroids = []
    liste_distance_intra_cluster = []
    # liste_centroids.append(centroids)

    while not converge and iteration <= max_iterations:
        assign_cluster = assignToCluster(data, centroids)


        computed_centroids = computeCentroid(data, assign_cluster)
        liste_centroids.append(computed_centroids)
        print("\n\n-----------------------------------------------------------------------------------------------------")
        print("Computed centroids :")
        print(computed_centroids.collect())
        print("-----------------------------------------------------------------------------------------------------\n\n")

        intra_cluster_distance = computeIntraClusterDistance(assign_cluster)
        print("\n\n-----------------------------------------------------------------------------------------------------")
        print("Distance Intra Cluster = " + str(intra_cluster_distance))
        print("-----------------------------------------------------------------------------------------------------\n\n")
        liste_distance_intra_cluster.append(intra_cluster_distance)

        if iteration > 0: # compte le nombre de points du nouveau rdd avec un cluster différent de l'iteration précédente
            moved = new_points.join(assign_cluster) \
                .filter(lambda l: l[1][0][0] != l[1][1][0]) \
                .count()
        # si pas de mouvement ou nombre max d'iterations atteint ou que distance intra cluster augmente alors il y a convergence
        if moved == 0 or iteration == max_iterations: #or (len(liste_distance_intra_cluster) >1 and liste_distance_intra_cluster[-2] < liste_distance_intra_cluster[-1]):
            converge = True
        else:
            centroids = computed_centroids
            new_points = assign_cluster
            iteration += 1

    # if liste_distance_intra_cluster[-2] < liste_distance_intra_cluster[-1]:
    #     return(liste_distance_intra_cluster[-2], liste_centroids[-2].collect(), iteration-1, assign_cluster)
    # else:
    return (intra_cluster_distance, centroids.collect(), iteration, assign_cluster)


if __name__ == "__main__":

    start_time = time.time()

    if len(sys.argv) != 4:
        print("Methode: kmeans.py file cluster max_iteration")
        exit(-1)

    # Create Spark conf
    conf = SparkConf().setAppName("kmeans").setMaster("local")
    sc = SparkContext(conf=conf)

    path = sys.argv[1]
    clusters = int(sys.argv[2])
    max_iterations = int(sys.argv[3])
    distances = []

    print("\n\n-----------------------------------------------------------------------------------------------------")
    solution = kmeans(sc, path, clusters, max_iterations)
    print("Nombre d'iterations : "+str(solution[2]))
    print("Distance intra cluster : "+str(solution[0]))
    print("Centroids : ")
    for i in solution[1]:
        print("Centroid : "+str(i[0])+"\t"+str(i[1]))
    print("-----------------------------------------------------------------------------------------------------\n\n")

    export_assign_data("mapped_data.csv", solution[3])

    ###### Retirer les commentaire suivants pour executer plusieurs kmeans et ainsi obtenir les moyennes
    # for i in range(0, max_iterations):
    #     print("\n\n-----------------------------------------------------------------------------------------------------")
    #     print("Iteration number: " + str(i))
    #     solution = kmeans(sc, path, clusters, max_iterations)
    #     distances.append(solution[0])
    #     print(solution)
    #     print("-----------------------------------------------------------------------------------------------------\n\n")
    #
    # print("Distance Intra cluster moyenne: " + str(np.mean(distances)))
    # print("Ecart type: " + str(np.std(distances)))
    #
    print("---- %s seconds ----" % (time.time() - start_time))

# sc = SparkContext()
# path_t = "C:/Users/Jeremy/Desktop/Fichier_Test_Generator_Py.csv"
#
# test = kmeans(sc, "Seasons_Stats.csv", 2, 50)
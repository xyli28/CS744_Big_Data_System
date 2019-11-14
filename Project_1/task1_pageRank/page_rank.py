import sys
import re
from operator import add
from pyspark.sql import SparkSession

def parse_line(line):
    """read a line from text and split to tokens by tab and blank"""
    tup = re.split(r'\s+', line)
    return tup[0].lower(), tup[1].lower()

def calculate_contribution(nodes, rank):
    """used to iteratively compute contribution from neighbors"""
    num_neighbors = len(nodes)
    for node in nodes:
        yield (node, rank / num_neighbors)

if __name__ == "__main__":

    # initialize spark app
    spark = SparkSession.builder.appName("PageRank").getOrCreate()
    # load file into RDD
    lines = spark.read.text(sys.argv[1]).rdd.map(lambda l: l[0])
    # coalesce lines by key
    node_neighbors_pairs = lines \
                        .map(lambda pair: parse_line(pair)) \
                        .distinct() \
                        .groupByKey() \
    # initialize rank of each key to 1.0
    node_ranks_pairs = node_neighbors_pairs.map(lambda node: (node[0], 1.0))
    # iteratively compute rank of each url 10 times
    for _ in range(10):
        contributions = node_neighbors_pairs \
            .join(node_ranks_pairs) \
            .flatMap(lambda rank: calculate_contribution(rank[1][0], rank[1][1]))
        node_ranks_pairs = contributions \
            .reduceByKey(add) \
            .mapValues(lambda rank: rank * 0.85 + 0.15)

    # combine partitions together and save to output path
    node_ranks_pairs.coalesce(1).saveAsTextFile(sys.argv[2])
    # terminate spark app
    spark.stop()

from pyspark.sql import SparkSession
from sys import argv

if __name__ == "__main__":
    # initialize spark app
    spark = SparkSession.builder.appName("sort").getOrCreate()
    # Read from HDFS
    df = spark.read.csv(argv[1], header = True)
    # Sort
    sorted_df = df.sort("cca2", "timestamp")
    # Write data to HDFS
    sorted_df.coalesce(1).write.csv(argv[2])
    # terminate spark app
    spark.stop()
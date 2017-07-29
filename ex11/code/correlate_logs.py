import sys
from pyspark.sql import SparkSession, functions, types, Row
import re
import math

spark = SparkSession.builder.appName('correlate logs').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+

line_re = re.compile("^(\\S+) - - \\[\\S+ [+-]\\d+\\] \"[A-Z]+ \\S+ HTTP/\\d\\.\\d\" \\d+ (\\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        return Row(hostname=m.group(1), transferred = m.group(2))
    else:
        return None

def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    # TODO: return an RDD of Row() objects
    return log_lines.map(line_to_row).filter(not_none)


def main():
    in_directory = sys.argv[1]
    
    logs = spark.createDataFrame(create_row_rdd(in_directory)).cache()
    
    log_by_hostname = logs.groupby("hostname").agg(functions.count("hostname"),functions.sum("transferred"))

    df_stat = log_by_hostname.groupby().agg(
        functions.count("hostname").alias("n"),
        functions.sum("count(hostname)").alias("x_sum"),
        functions.sum("sum(transferred)").alias("y_sum"),
        functions.sum(log_by_hostname["count(hostname)"]**2).alias("x_sq_sum"),
        functions.sum(log_by_hostname["sum(transferred)"]**2).alias("y_sq_sum"),
        functions.sum(log_by_hostname["count(hostname)"] * log_by_hostname["sum(transferred)"]).alias("xy_sum")
    )
    
    stat = df_stat.first()
    r = (stat.n * stat.xy_sum - stat.x_sum * stat.y_sum) / (math.sqrt(stat.n * stat.x_sq_sum - stat.x_sum**2) * math.sqrt(stat.n * stat.y_sq_sum - stat.y_sum**2 ))
#     r = 0 # TODO: it isn't zero.
    print("r = %g\nr^2 = %g" % (r, r**2))


if __name__=='__main__':
    main()


import sys
from pyspark.sql import SparkSession, functions, types, Row
import re

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
        # TODO
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


def main():
    in_directory = sys.argv[1]
    logs = spark.createDataFrame(create_row_rdd(in_directory))

    # TODO: calculate r.

    r = 0 # TODO: it isn't zero.
    print("r = %g\nr^2 = %g" % (r, r**2))


if __name__=='__main__':
    main()
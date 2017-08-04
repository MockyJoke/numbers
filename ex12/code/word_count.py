import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+

import string, re
wordbreak = r'[%s\s]+' % (re.escape(string.punctuation),)  # regex that matches spaces and/or punctuation

def main():
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    
    words = spark.read.text(in_directory)
    words.show()
    sperated = words.select(functions.explode(functions.split(functions.lower(words["value"]),wordbreak)).alias("word"))
#     sperated.show()
    word_count = sperated.groupby("word").agg(functions.count("word").alias("count"))
#     word_count.show()
    sorted_word_count = word_count.orderBy(functions.desc("count"),"word").filter(word_count["word"] !="")
    
    sorted_word_count.write.csv(out_directory, mode = "overwrite")
if __name__=='__main__':
    main()


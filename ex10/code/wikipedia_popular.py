
# coding: utf-8

# In[25]:

import findspark
findspark.init()

import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('wikipedia_popular').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+


schema = types.StructType([ # commented-out fields won't be read
    #types.StructField('archived', types.BooleanType(), False),
    #types.StructField('author', types.StringType(), False),
    #types.StructField('author_flair_css_class', types.StringType(), False),
    #types.StructField('author_flair_text', types.StringType(), False),
    #types.StructField('body', types.StringType(), False),
    #types.StructField('controversiality', types.LongType(), False),
    #types.StructField('created_utc', types.StringType(), False),
    #types.StructField('distinguished', types.StringType(), False),
    #types.StructField('downs', types.LongType(), False),
    #types.StructField('edited', types.StringType(), False),
    #types.StructField('gilded', types.LongType(), False),
    #types.StructField('id', types.StringType(), False),
    #types.StructField('link_id', types.StringType(), False),
    #types.StructField('name', types.StringType(), False),
    #types.StructField('parent_id', types.StringType(), True),
    #types.StructField('retrieved_on', types.LongType(), False),
    #types.StructField('score', types.LongType(), False),
    #types.StructField('score_hidden', types.BooleanType(), False),
    #types.StructField('subreddit', types.StringType(), False),
    #types.StructField('subreddit_id', types.StringType(), False),
    #types.StructField('ups', types.LongType(), False),
    
    types.StructField('lang', types.StringType(), False),
    types.StructField('title', types.StringType(), False),
    types.StructField('visit_count', types.LongType(), False),
    types.StructField('data_size', types.LongType(), False),
])

def pathToTime(path):
    start = path.rfind("pagecounts-") + 11
    end = path.rfind(".")-4
    return path[start:end]

def main():
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    # in_directory = "pagecounts-1"
    # out_directory = "output"
    pagecounts = spark.read.csv(in_directory,sep = " ", schema = schema).withColumn('filename', functions.input_file_name())
    pagecounts = pagecounts.drop("data_size")
    pagecounts = pagecounts.filter((pagecounts["lang"]=="en") &
                                   (pagecounts["title"] != "Main_Page") &
                                   ((pagecounts["title"].startswith("Special:")==False)))
    # pagecounts.show()
    path_to_hour = functions.udf(pathToTime, returnType=types.StringType())
    pagecounts = pagecounts.withColumn("time",path_to_hour("filename")).cache()
    
    
    max_visit_counts = pagecounts.groupby("time").agg(functions.max("visit_count"))
    
    # adapted from https://spark.apache.org/docs/2.0.0/api/python/pyspark.sql.html
    result = pagecounts.join(max_visit_counts,(pagecounts["time"] == max_visit_counts["time"]))
    result = result.filter(result["visit_count"]== result["max(visit_count)"])
    result = result.select(pagecounts["time"],"title","visit_count").sort("time")
    # result.explain()

    # We know groups has <=100 rows, since the data is grouped by time combincations, so it can safely be moved into 1 partition.
    result.coalesce(1).write.csv(out_directory, mode='overwrite')



if __name__=='__main__':
    main()


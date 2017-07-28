import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+

schema = types.StructType([ # commented-out fields won't be read
    #types.StructField('archived', types.BooleanType(), False),
    types.StructField('author', types.StringType(), False),
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
    types.StructField('score', types.LongType(), False),
    #types.StructField('score_hidden', types.BooleanType(), False),
    types.StructField('subreddit', types.StringType(), False),
    #types.StructField('subreddit_id', types.StringType(), False),
    #types.StructField('ups', types.LongType(), False),
])


def main():
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    
    comments = spark.read.json(in_directory, schema=schema).cache()
    # comments.show()

    averages_by_subreddit = comments.groupby("subreddit").agg(functions.avg("score").alias("avg(score)"))
    averages_by_subreddit = averages_by_subreddit.where(averages_by_subreddit["avg(score)"] > 0)
    # averages_by_subreddit.show()
    averages_by_subreddit = functions.broadcast(averages_by_subreddit)
    comments_with_avg = comments.join(averages_by_subreddit, "subreddit")
    comments_with_avg = comments_with_avg.withColumn("rel_score", comments_with_avg["score"] / comments_with_avg["avg(score)"]).cache()
    # comments_with_avg.show()

    max_relative_score = comments_with_avg.groupby(comments["subreddit"]).agg(functions.max("rel_score"))
    # max_relative_score.show()
    
    max_relative_score = functions.broadcast(max_relative_score)
    comments_with_max = comments_with_avg.join(max_relative_score,"subreddit")
    # comments_with_max.show()
    max_comments = comments_with_max.where(comments_with_max["max(rel_score)"]==comments_with_max["rel_score"])

    best_author = max_comments.select(max_comments["subreddit"], max_comments["author"], max_comments["rel_score"])
    
    best_author.write.json(out_directory, mode='overwrite')


if __name__=='__main__':
    main()


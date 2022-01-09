import org.apache.spark.sql.SparkSession

val ss = SparkSession.builder()
                .master("local[*]")
                .appName("KMeans")
                //.config("spark.driver.memory", "4g")
                .getOrCreate

ss.sparkContext.getConf.getAll

val inputFile = "/home/panos/Downloads/Greek_Parliament_Proceedings_1989_2020_DataSample.csv"
val stopwords_gr = "stopwords.txt"

val df = ss.read.option("header", true)
            .csv(inputFile)
            .select("member_name", "sitting_date", "speech")
            .na
            .drop

df.printSchema

// keep rows of a specific year
import org.apache.spark.sql.functions.{to_date, to_timestamp}

val df_date = df.withColumn("date_y", to_date($"sitting_date", "dd/MM/yyyy")).drop("sitting_date")

df_date.printSchema

// val year = 2020 args(0).toInt

val speechesDF = df_date.groupBy("member_name")
                    .agg(concat_ws(",", collect_list("speech")).as("speeches"))

val cleanSpeechesDF = speechesDF.withColumn("speechesClean", regexp_replace($"speeches", "[\\_,\\*,\\$,\\#,\\@,\\&]", ""))

// cleanSpeechesDF.show

import org.apache.spark.ml.feature.RegexTokenizer

val speechesDF_tok = new RegexTokenizer().setInputCol("speechesClean")
                                            .setOutputCol("speechesTok")
                                            .setMinTokenLength(4)
                                            .setToLowercase(true)
                                            .setPattern("[\\s.,!-~'\";*^%$@()&<>/+_ ]")
                                            .transform(cleanSpeechesDF)

// speechesDF_tok.show

import ss.implicits._

// Filter Stopwords with spark StopWordsRemover
import org.apache.spark.ml.feature.StopWordsRemover

val stopwords : Array[String] = sc.textFile(stopwords_gr).collect.toSet.toArray

val swr = new StopWordsRemover().setCaseSensitive(false)
                            .setInputCol("speechesTok")
                            .setOutputCol("speechesTok1")
                            .setLocale("el")
                            .setStopWords(stopwords)

val speechesFilteredDF = swr.transform(speechesDF_tok)

// speechesFilteredDF.show(50)

import org.apache.spark.ml.feature.{CountVectorizerModel, CountVectorizer}

val cvModel : CountVectorizerModel = new CountVectorizer().setInputCol("speechesTok1")
                                        .setOutputCol("rawFeatures")
                                        .setMaxDF(250) 
                                        .setVocabSize(100000)
                                        .fit(speechesFilteredDF)


val cvDF1 = cvModel.transform(speechesFilteredDF).drop("speeches", "speechesClean", "speechesTok")

// cvDF1.show

import org.apache.spark.ml.feature.IDF

val idfModel = new IDF().setInputCol("rawFeatures")
                        .setOutputCol("features")
                        .fit(cvDF1)

val cvDF = idfModel.transform(cvDF1)

// cvDF.show(10)

import org.apache.spark.ml.linalg.Vector 
import org.apache.spark.rdd.RDD

val n_most_freq = 40

val zippedVoc = cvModel.vocabulary.zipWithIndex

val mostFreq_rdd : RDD[Array[String]]  = cvDF.select("features")
                .rdd
                .map(_.getAs[Vector](0))
                .map(_.toSparse)
                .map{ row => 
                        row.indices.zip(row.values)
                            .sortBy(_._2).take(n_most_freq).map(_._1) }
                .map(arr => {
                        
                        zippedVoc.map { case (word, idx) => 
                            if (arr contains idx) 
                                word.toString
                        }
                    }
                .filter(_.!=()))
                .map(arr => arr.map(_.toString))

// mostFreq_rdd.take(50)

import org.apache.spark.sql.expressions.Window 

val members = speechesDF.select("member_name").rdd.map(w => w.toString.replaceAll("[\\[\\]]","").capitalize).toDF("name").withColumn("id", row_number().over(Window.orderBy("name"))).cache()

val df2 = mostFreq_rdd.toDF("Most_Frequent")

// df2.show(30)

val mostFreqDF = df2.withColumn("id", row_number().over(Window.orderBy("Most_Frequent")))

// mostFreqDF.show

// members.show

val finalDF = members.join(mostFreqDF, "id").drop("id")

// finalDF.show(20, false)

import org.apache.spark.ml.feature.Word2Vec

val word2vecDF = new Word2Vec().setMaxSentenceLength(50)
                             .setMinCount(0)
                             .setInputCol("Most_Frequent")
                             .setOutputCol("embeddings")
                             .setVectorSize(5)
                             .fit(finalDF)
                             .transform(finalDF)


import org.apache.spark.ml.clustering.KMeans

val kmeansDF = new KMeans().setFeaturesCol("embeddings")
                            .setSeed(42)
                            .setK(3)
                            .setMaxIter(5)
                            .fit(word2vecDF)
                            .transform(word2vecDF)

kmeansDF.show(10)

kmeansDF.select("name", "prediction").write.option("header", true).csv("kmeans_out")

import scala.collection.mutable.WrappedArray


finalDF.rdd.
        map { r : org.apache.spark.sql.Row => 
            (r.getAs[String](0), s"(${year},(" + (
                r.getAs[WrappedArray[String]](1).mkString(",").toString) + ")")
                }.saveAsTextFile(s"results_${year}")


/*

finalDF.rdd.
    map { r : org.apache.spark.sql.Row => 
        ((r.getAs[String](0), Array((year : Int, Array( 
            r.getAs[WrappedArray[String]](1).toArray.mkString(","))))))}.take(1)//.saveAsTextFile("test3")
*/

val x = sc.textFile("results_2015/part-00000").map(x => x.split(",")).map(x => x(1)).collect
                    


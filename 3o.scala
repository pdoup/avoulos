val df = spark.read.option("header", true)
            .csv("/home/panos/Downloads/Greek_Parliament_Proceedings_1989_2020_DataSample.csv")
            .select("member_name", "speech")
            .na
            .drop

val speechesDF = df.groupBy("member_name")
                    .agg(concat_ws(",", collect_list("speech")).as("speeches"))

val cleanSpeechesDF = speechesDF.withColumn("speechesClean", regexp_replace($"speeches", "[\\_,\\*,\\$,\\#,\\@,\\&]", ""))

cleanSpeechesDF.show

import org.apache.spark.ml.feature.RegexTokenizer

val speechesDF_tok = new RegexTokenizer().setInputCol("speechesClean")
                                            .setOutputCol("speechesTok")
                                            .setMinTokenLength(4)
                                            .setToLowercase(true)
                                            .setPattern("[\\s.,!-~'\";*^%$@()&<>/+_ ]")
                                            .transform(cleanSpeechesDF)

speechesDF_tok.show

val stopwords : Set[String] = sc.textFile("stopwords.txt").collect.toSet[String]

import spark.implicits._

val filter_stopwords_udf = udf ( (v : scala.collection.mutable.WrappedArray[String]) => v.filterNot(w => stopwords contains w) )

val speechesFilteredDF = speechesDF_tok.withColumn("speechesTok1", filter_stopwords_udf(speechesDF_tok("speechesTok")))

speechesFilteredDF.show

import org.apache.spark.ml.feature.{CountVectorizerModel, CountVectorizer}

val cvModel : CountVectorizerModel = new CountVectorizer().setInputCol("speechesTok1")
                                        .setOutputCol("features")
                                        .setMinTF(2)
                                        .setMaxDF(10) 
                                        .setVocabSize(10)
                                        .fit(speechesFilteredDF)


val cvDF = cvModel.transform(speechesFilteredDF).drop("speeches", "speechesClean", "speechesTok")

cvDF.show

import org.apache.spark.ml.linalg.Vector 
import org.apache.spark.rdd.RDD

val n_most_freq = 5

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
                            if (arr.contains(idx)) 
                                word.toString
                        }
                    }
                .filter(_.!=()))
                .map(arr => arr.map(_.toString))
                

mostFreq_rdd.take(5)

import org.apache.spark.sql.expressions.Window 

val members = speechesDF.select("member_name").rdd.map(w => w.toString.replaceAll("[\\[\\]]","").capitalize).toDF("name").withColumn("id", row_number().over(Window.orderBy("name"))).cache()

val year = 2020

val df2 = mostFreq_rdd.toDF(s"Most_Frequent_${year}")

df2.show(30)

val mostFreqDF = df2.withColumn("id", row_number().over(Window.orderBy(s"Most_Frequent_${year}")))

mostFreqDF.show

members.show

val finalDF = members.join(mostFreqDF, "id").drop("id")

finalDF.show(10, false)

import scala.collection.mutable.WrappedArray

finalDF.rdd.map { r : org.apache.spark.sql.Row => 
    (r.getAs[String](0), "(", (r.getAs[WrappedArray[String]](1).mkString(",")), ")") }.saveAsTextFile("ssladja12asd") 



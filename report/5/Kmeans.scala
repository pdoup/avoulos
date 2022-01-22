import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, CountVectorizerModel, CountVectorizer, IDF, Word2Vec}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.commons.lang3.StringUtils

object Kmeans {

  val inputFile = "file:///home/ozzy/Desktop/bd/dtst.csv"
  val stopwords_gr = "file:///home/ozzy/Desktop/bd/avoulos-main/aux_files/new_stop.txt"

  def main(args : Array[String]) : Unit = {

    val n_most_freq = args(0).toInt
    val k = args(1).toInt

    val ss = SparkSession.builder()
      .appName("KMeans")
      .getOrCreate

    import ss.implicits._

    ss.sparkContext.setLogLevel("WARN")

    val df = ss.read.option("header", true)
      .csv(inputFile)
      .select("political_party", "sitting_date", "speech")
      .na
      .drop
      .withColumn("date_y", to_date($"sitting_date", "dd/MM/yyyy"))
      .drop("sitting_date").sample(false, 0.4f, 42L).cache()



    	val check = udf((colValue: String) => {StringUtils.stripAccents(colValue)})
	val df15 = df.select($"political_party",$"date_y", check($"speech").as("speech"))


    val concatSpeechesDF = df15.groupBy("political_party")
      .agg(concat_ws(",", collect_list("speech"))
        .as("speeches"))

    val speechesDF = new RegexTokenizer().setInputCol("speeches")
      .setOutputCol("speeches_tok")
      .setMinTokenLength(4)
      .setToLowercase(true)
      .setPattern("[\\s.,!-~'…\"’΄;*^%$@«?|»{}()&–<>/+_ ]")
      .transform(concatSpeechesDF)


    val stopwords : Array[String] = ss.sparkContext.textFile(stopwords_gr)
      .collect
      .toSet
      .toArray

    val speechesFilteredDF = new StopWordsRemover().setCaseSensitive(false)
      .setInputCol("speeches_tok")
      .setOutputCol("speeches_tok1")
      .setLocale("el")
      .setStopWords(stopwords)
      .transform(speechesDF)


   

    val cvModel : CountVectorizerModel = new CountVectorizer().setInputCol("speeches_tok1")
      .setOutputCol("rawFeatures")
      .setMaxDF(0.4f)
      .setVocabSize(70000)
      .fit(speechesFilteredDF)

    val cvModelDF = cvModel.transform(speechesFilteredDF)
      .drop("speeches", "speeches_tok", "speeches_tok1")


    val cvDF = new IDF().setInputCol("rawFeatures")
      .setOutputCol("features")
      .fit(cvModelDF)
      .transform(cvModelDF)


    val zippedVoc = cvModel.vocabulary.zipWithIndex

    val mostFreq_rdd : RDD[Array[String]]  = cvDF.select("features")
      .rdd
      .map(_.getAs[Vector](0))
      .map(_.toSparse)
      .map{ row =>
        row.indices.zip(row.values)
          .sortBy(_._2).take(n_most_freq).map(_._1) }
      .map( arr => {

        zippedVoc.map { case (word, idx) =>
          if (arr contains idx)
            word.toString
        }
      }
        .filter(_.!=()))
      .map(arr => arr.map(_.toString))


    val membersDF = concatSpeechesDF.select("political_party")
      .rdd
      .map(w => w.toString.replaceAll("[\\[\\]]","").capitalize)
      .toDF("name")
      .withColumn("id", row_number().over(Window.orderBy("name")))



    val mostFreqDF = mostFreq_rdd.toDF("Most_Frequent_TFIDF")
      .withColumn("id", row_number().over(Window.orderBy("Most_Frequent_TFIDF")))


    val finalDF = membersDF.join(mostFreqDF, "id").drop("id")


    val word2vecDF = new Word2Vec().setMaxSentenceLength(n_most_freq)
      .setMinCount(0)
      .setInputCol("Most_Frequent_TFIDF")
      .setOutputCol("embeddings")
      .setVectorSize(100)
      .fit(finalDF)
      .transform(finalDF)

    val kmeansDF = new KMeans().setFeaturesCol("embeddings")
      .setSeed(42)
      .setK(k)
      .setMaxIter(100)
      .fit(word2vecDF)
      .transform(word2vecDF)
      
      
    println("=============================================")
    println("Evaluating the clustering w/ silhouette score...")


    val silhouette = new ClusteringEvaluator().setFeaturesCol("embeddings").evaluate(kmeansDF)
    println(f"score : $silhouette%.7f")

    println("=============================================")

    kmeansDF.groupBy("prediction").count().show(false)

    kmeansDF.select("name", "prediction").write.option("header", true).csv(s"file:///home/ozzy/Desktop/bd/Erwthma5/kmeans_out_k_party${k}")

    println("Output written to csv file")

    ss.stop()
  }
}

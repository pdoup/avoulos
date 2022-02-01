import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, CountVectorizerModel, CountVectorizer, IDF, Word2Vec}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.commons.lang3.StringUtils

// Clustering members or parties using their most represatative keywords

/* 
   NOTE : This code is meant to run for parties only since the aggregation is done by parties. 
   	  To alter this so that it can be executed with members, all the instances of political_party
	  can be changed to member_name. In the zip file provided there are results for both members and parties
*/

// Takes 2 command line parameters
// n_most_freq : number of most important words to take into account
// k : number of clusters for kmeans

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
	
    // read the csv and convert the date to date format, also take a sample so that it is less computationally intensive
    val df = ss.read.option("header", true)
      .csv(inputFile)
      .select("political_party", "sitting_date", "speech")
      .na
      .drop
      .withColumn("date_y", to_date($"sitting_date", "dd/MM/yyyy"))
      .drop("sitting_date").sample(false, 0.4f, 42L).cache()


	// strip accents
    	val check = udf((colValue: String) => {StringUtils.stripAccents(colValue)})
	val df15 = df.select($"political_party",$"date_y", check($"speech").as("speech"))

    
    // group by political_party or member
    val concatSpeechesDF = df15.groupBy("political_party")
      .agg(concat_ws(",", collect_list("speech"))
        .as("speeches"))

    // tokenization
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

    // removing stopwords
    val speechesFilteredDF = new StopWordsRemover().setCaseSensitive(false)
      .setInputCol("speeches_tok")
      .setOutputCol("speeches_tok1")
      .setLocale("el")
      .setStopWords(stopwords)
      .transform(speechesDF)

    // build a count vectorizer model TF
    val cvModel : CountVectorizerModel = new CountVectorizer().setInputCol("speeches_tok1")
      .setOutputCol("rawFeatures")
      .setMaxDF(0.4f)
      .setVocabSize(70000)
      .fit(speechesFilteredDF)

    val cvModelDF = cvModel.transform(speechesFilteredDF)
      .drop("speeches", "speeches_tok", "speeches_tok1")

    // IDF
    val cvDF = new IDF().setInputCol("rawFeatures")
      .setOutputCol("features")
      .fit(cvModelDF)
      .transform(cvModelDF)


    val zippedVoc = cvModel.vocabulary.zipWithIndex

    // take the n most frequent keywords per member or per party
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

    // train a w2vec model on the n most frequent words which will serve as input for the kmeans
    val word2vecDF = new Word2Vec().setMaxSentenceLength(n_most_freq)
      .setMinCount(0)
      .setInputCol("Most_Frequent_TFIDF")
      .setOutputCol("embeddings")
      .setVectorSize(100)
      .fit(finalDF)
      .transform(finalDF)

    // K-Means to find meaningful clusters using w2vec representations
    // k is a command line parameter
    val kmeansDF = new KMeans().setFeaturesCol("embeddings")
      .setSeed(42)
      .setK(k)
      .setMaxIter(100)
      .fit(word2vecDF)
      .transform(word2vecDF)
      
      
    println("=============================================")
    println("Evaluating the clustering w/ silhouette score...")

    // evaluate the clustering w/ silhouette
    val silhouette = new ClusteringEvaluator().setFeaturesCol("embeddings").evaluate(kmeansDF)
    println(f"score : $silhouette%.7f")

    println("=============================================")

    kmeansDF.groupBy("prediction").count().show(false)
    // write to file the name of the member/party and the cluster in belongs to
    kmeansDF.select("name", "prediction").write.option("header", true).csv(s"file:///home/ozzy/Desktop/bd/Erwthma5/kmeans_out_k_party${k}")

    println("Output written to csv file")

    ss.stop()
  }
}

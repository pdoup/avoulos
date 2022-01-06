import org.apache.spark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, HashingTF, IDF, RegexTokenizer}
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.ml.linalg.{Vectors => NewVectors}
import org.apache.spark.rdd.RDD
import scala.collection.mutable


// Program takes 1 parameter as argument (K), the number of top K most similar members

object PairWise {


  def main(args: Array[String]): Unit = {

    // Create a new spark session
    val ss = SparkSession.builder().appName("lda").master("local[*]").getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")
    import ss.implicits._

    val df = ss.read.option("header", true)
      .text("file:///home/ozzy/Desktop/bd/dtst.csv")
      .select("speech")
      .na
      .drop

    df.printSchema

    val cleanSpeechesDF = df.withColumn("speechesClean", regexp_replace($"speech", "[\\_,\\*,\\$,\\#,\\@,\\&]", ""))

    import org.apache.spark.ml.feature.RegexTokenizer

    val speechesDF_tok = new RegexTokenizer().setInputCol("speechesClean")
      .setOutputCol("speechesTok")
      .setMinTokenLength(4)
      .setToLowercase(true)
      .setPattern("[\\s.,!-~'\";*^%$@()&<>/+_ ]")
      .transform(cleanSpeechesDF)

    speechesDF_tok.show

    val stopwordsPath : String = "file:///home/ozzy/Desktop/bd/avoulos-main/aux_files/stopwords_gr.txt"

    val stopwords = ss.sparkContext.textFile(stopwordsPath)
      .map(w => w.dropRight(2))
      .collect
      .toSet


    val filter_stopwords_udf = udf ( (v : scala.collection.mutable.WrappedArray[String]) => v.filterNot(w => stopwords contains w) )

    val speechesFilteredDF = speechesDF_tok.withColumn("speechesTok1", filter_stopwords_udf(speechesDF_tok("speechesTok")))

    val prehash = speechesFilteredDF.drop("speech", "speechesClean", "speechesTok")

  //######################################################################HASHING###########################
    import org.apache.spark.ml.feature.{HashingTF,IDF}
    val kmeansDF = new HashingTF()
      .setInputCol("speechesTok1").setOutputCol("rawFeatures").setNumFeatures(10).transform(prehash)

    kmeansDF.show


    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(kmeansDF)
    val rescaledData = idfModel.transform(kmeansDF)



    import org.apache.spark.ml.clustering.KMeans
    import org.apache.spark.ml.evaluation.ClusteringEvaluator


    // Trains a k-means model.
    val kmeans = new KMeans().setK(8).setSeed(1L).setFeaturesCol("features")
    val model = kmeans.fit(rescaledData)

    // Make predictions
    val predictions = model.transform(rescaledData)

    // Evaluate clustering by computing Silhouette score
    val evaluator = new ClusteringEvaluator()

    val silhouette = evaluator.evaluate(predictions)
    println(s"Silhouette with squared euclidean distance = $silhouette")

    // Shows the result.
    //println("Cluster Centers: ")
    //model.clusterCenters.foreach(println)

    predictions.groupBy("prediction").count().show



    ss.stop()
  }
}

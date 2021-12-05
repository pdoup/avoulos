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

  final val inputPath : String = "file:///home/ozzy/path/to/csv"
  final val stopwordsPath : String = "file:///home/ozzy/path/to/stopwords"

  def main(args: Array[String]): Unit = {

    val ss = SparkSession.builder().appName("pairwise").master("local[*]").getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")

    import ss.implicits._

    // load csv file as a data frame
    val df = ss.read.option("header", "true")
      .csv(inputPath)
      .select($"member_name",$"speech")
      .na.drop()

    // concat all speeches grouped by each candidate
    val speechesDF = df.groupBy("member_name")
      .agg(concat_ws(",", collect_list("speech")).alias("speeches"))

    // tokenize speeches
    val speechesTokenizedDF = new RegexTokenizer().setInputCol("speeches")
      .setOutputCol("speechesTok")
      .setMinTokenLength(4)
      .setToLowercase(true)
      .setPattern("[\\s.,!-~'\";*^%#$@()&<>/ ]")
      .transform(speechesDF)

    // read stopwords file
    val stopwords_gr = ss.sparkContext.textFile(stopwordsPath)
      .map(w => w.dropRight(2))
      .collect
      .toSet

    // user-defined-function to perform the filtering
    val udf_filter_stop = udf ( (s : mutable.WrappedArray[String]) => s.filter(w => !stopwords_gr.contains(w)))

    // apply the stopwords filter
    val cleanDF = speechesTokenizedDF.
      select($"member_name".as("id"), udf_filter_stop($"speechesTok").as("rFeatures"))

    // TF-IDF pipeline

    // First use hashing tf to get the term frequencies
    val hashingTF = new HashingTF()
      .setNumFeatures(50000)
      .setInputCol("rFeatures")
      .setOutputCol("rawFeatures")

    // then apply the TF-IDF
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

    // create the pipeline and run it
    val speechesTF_IDF = new Pipeline()
      .setStages(Array(hashingTF, idf))
      .fit(cleanDF)
      .transform(cleanDF)
      .drop(Seq("rawFeatures", "rFeatures") : _*)

    // user-defined function to convert tf-idf to a dense vector
    val udf_toDense = udf((s : org.apache.spark.ml.linalg.Vector) => s.toDense)

    // apply the udf to the rFeatures column (TF-IDF) and return a single column data frame with the TF-IDF scores
    val df1 = speechesTF_IDF.withColumn("dense", udf_toDense($"features")).select("dense")

    /*
        create a new dataframe with 2 columns, the index (in ascending order from 1 to length.rows)
        and the TF-IDF scores
    */
    val df2 = df1.withColumn("indices", row_number().over(Window.orderBy("dense")))

    // reverse the order of the columns so that index is first
    val df3 = df2.select(df2.columns.map(df2(_)).reverse : _*)

    // create an indexed row matrix and transpose it so that each column holds the TF-IDF score for a member
    val rowMatrix = new IndexedRowMatrix(df3.rdd.map(r =>
      IndexedRow(r(0).asInstanceOf[Number].longValue(),
        OldVectors.fromML(r(1).asInstanceOf[org.apache.spark.ml.linalg.Vector]))))
      .toBlockMatrix
      .transpose
      .toIndexedRowMatrix

    // once the matrix is transpose we can compute column-column cosine similarities for each pair of columns
    val similaritiesDF = rowMatrix.columnSimilarities().entries.toDF(Seq("S1","S2","CosineSim") : _*).cache()

    // create an indexed data frame, the first column is the ascending index and the second the name of the member(id)
    val namesDF = cleanDF.withColumn("index", row_number().over(Window.orderBy("rFeatures")))
      .select($"index", $"id")

    // convert the namesDF to a map of key-value pairs (index, member_name)
    val mapToNames : Map[Int,String] = namesDF
      .map(row => (row(0).asInstanceOf[Number].intValue(),row(1).asInstanceOf[String]))
      .collect
      .toMap

    // udf to map an index to the member associated with it
    val udf_toName = udf( (s : Int) => mapToNames getOrElse(s,"null") )

    val reorderColumns : Seq[String] = Seq("member_name_1", "member_name_2", "CosineSim")
    // apply the udf to map the first two columns of integers to actual member names
    val finalDF = similaritiesDF
      .withColumn("member_name_1", udf_toName($"S1"))
      .withColumn("member_name_2", udf_toName($"S2"))
      .drop(Seq("S1", "S2") : _*)
      .select(reorderColumns.head, reorderColumns.tail : _*)

    // show top K most similar members
    val topK : Int = args(0).toInt
    finalDF.sort(desc("CosineSim")).show(topK,false)

    ss.stop()
  }
}

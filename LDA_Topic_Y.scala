import org.apache.spark.ml.clustering.{LDA, LDAModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, RegexTokenizer}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.to_date

import scala.collection.mutable


object LDA_Topic_Y {

  def showTopics(ldaM : LDAModel, vocab : Array[String], mTopics : Int = 10): Unit = {
    ldaM.describeTopics(maxTermsPerTopic = mTopics).collect().foreach
    { r => {
      println("Topic: " + r(0))
      val terms:mutable.WrappedArray[Int] = r(1).asInstanceOf[mutable.WrappedArray[Int]]
      terms.foreach {
        t => { println("Term: " + vocab(t)) }
      }
      println()
      }
    }
  }

  def main(args: Array[String]): Unit = {

    val inputPath = "file:///path/to/csv"
    val stopwordPath : String = "file:///path/to/stopwords/txt"

    // Create a new spark session
    val ss = SparkSession.builder().appName("lda_year").master("local[*]").getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")

    // load csv file as a data frame
    val df = ss.read.option("header", "true").csv(inputPath)

    // Drop rows that contain NA's
    val df1 = df.na.drop()

    import ss.implicits._

    // convert sitting_date column to date format
    val df2 = df1.withColumn("sitting_date", to_date($"sitting_date", "dd/MM/yyyy"))

    // df2.printSchema

    // create new column speechTok with the tokenized speeches using regex
    val df3 = new RegexTokenizer().setInputCol("speech")
      .setOutputCol("speechTok")
      .setMinTokenLength(4) // reject tokens with length < 4
      .setToLowercase(true) // convert to lowercase
      .setPattern("[\\s.,!-~'\";*^%#$@()&<>/ ]")
      .transform(df2)

    val stopwords_gr = ss.sparkContext.textFile(stopwordPath)
      .map(w => w.dropRight(2))
      .collect
      .toSet

    // user-defined-function to perform the filtering
    val udf_filter_stop = udf ( (s : mutable.WrappedArray[String]) => s.filter(w => !stopwords_gr.contains(w)))

    // apply the stopwords filter
    val df4 = df3.select($"sitting_date", udf_filter_stop($"speechTok").as("rFeatures"))

    // create a count vectorizer model
    val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("rFeatures")
      .setOutputCol("features")
      .setVocabSize(50000)
      .fit(df4)

    // fit it on the data frame
    val df5 = cvModel.transform(df4)

    // create a temp view on the dataframe to execute sql queries on it
    df5.createOrReplaceTempView("date_df")

    // sql statement to return the speeches of each year or a range of years
    // store the result in an array of data frames
    var df_years : Seq[DataFrame] = Seq()
    for (year <- 1989 to 2020) {
      df_years = df_years :+ ss.sql(s"select features from date_df where year(sitting_date) = ${year}")
    }
    
    val k = 3
    val maxIter = 10

    // run the LDA algorithm for a specific year
    // for year 1990 -> df_years(1)
    val ldaModel = new LDA().setK(k).setMaxIter(maxIter).fit(df_years(1))

    // print all the topics and the top k words associated with that topic
    val maxWordsPerTopic_ = 10
    // optionally you can specify the number of topics to show
    showTopics(ldaModel, cvModel.vocabulary)
  }
}

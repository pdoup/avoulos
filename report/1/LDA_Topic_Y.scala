import org.apache.spark.ml.clustering.{LDA, LDAModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, RegexTokenizer}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.to_date
import org.apache.commons.lang3.StringUtils

import scala.util.control.Breaks._

import scala.collection.mutable


object LDA_Topic_Y {
  
  // aux function to print the topics and the top M corresponding keywords per year
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

    val inputPath = "file:///home/ozzy/Desktop/bd/dtst.csv"
    val stopwordPath = "file:///home/ozzy/Desktop/bd/avoulos-main/aux_files/new_stop.txt"

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
	val check = udf((colValue: String) => {StringUtils.stripAccents(colValue)})
	val df25 = df2.select($"sitting_date", check($"speech").as("speech"))
 	df25.show()
    // create new column speechTok with the tokenized speeches using regex
    val df3 = new RegexTokenizer().setInputCol("speech")
      .setOutputCol("speechTok")
      .setMinTokenLength(4) // reject tokens with length < 4
      .setToLowercase(true) // convert to lowercase
      .setPattern("[\\s.,!-~'…\"’΄;*^%$@«?|»{}()&–<>/+_ ]")
      .transform(df25)

    val stopwords_gr = ss.sparkContext.textFile(stopwordPath)
      .collect
      .toSet

    // user-defined-function to perform the filtering
    val udf_filter_stop = udf ( (s : mutable.WrappedArray[String]) => s.filter(w => !stopwords_gr.contains(w)))

    // apply the stopwords filter
    val df4 = df3.select($"sitting_date", udf_filter_stop($"speechTok").as("rFeatures"))

    // create a count vectorizer model
    val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("rFeatures")
      .setOutputCol("features")
      .setVocabSize(100000)
      .fit(df4)

    // fit it on the data frame
    val df5 = cvModel.transform(df4)

    // create a temp view on the dataframe to execute sql queries on it
    df5.createOrReplaceTempView("date_df")

    // sql statement to return the speeches of each year or a range of years
    // store the result in an array of data frames
    var df_years : Seq[DataFrame] = Seq()
    for (year <- 1989 to 2020) {
      if (year != 1995) {
        df_years = df_years :+ ss.sql(s"select features from date_df where year(sitting_date) = ${year}")
	      
      }
    }
   // val yr = ss.sql(s"select features from date_df where year(sitting_date) = year")
    val k = 5
    val maxIter = 70
    val maxWordsPerTopic_ = 10

for (year <- 1989 to 2020) {
    breakable {
      if (year == 1995) {
        break  // break out of the 'breakable', continue the outside loop
      } 
      else {
        println("Year",year)
      var ldaModel = new LDA().setK(k).setMaxIter(maxIter).fit(ss.sql(s"select features from date_df where year(sitting_date) = $year"))

      // print all the topics and the top k words associated with that topic

      // optionally you can specify the number of topics to show
      showTopics(ldaModel, cvModel.vocabulary, maxWordsPerTopic_)
    }
    	}
		}
      			}
    				}
  



 

import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, RegexTokenizer}
import org.apache.spark.sql.functions.udf
import org.apache.commons.lang3.StringUtils

import scala.collection.mutable

object LDA_Topic {
  
  def main(args: Array[String]): Unit = {
    
  val inputPath = "file:///home/ozzy/Desktop/bd/dtst.csv"
  val stopwordPath = "file:///home/ozzy/Desktop/bd/avoulos-main/aux_files/new_stop.txt"
    
    // Create a new spark session
    val ss = SparkSession.builder().appName("lda").master("local[*]").getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")
	import ss.implicits._

    // load csv file as a data frame
    val df = ss.read.option("header", "true").csv(inputPath)

    // Drop rows that contain NA's
    val df1 = df.na.drop()


   	val check = udf((colValue: String) => {StringUtils.stripAccents(colValue)})
	val df15 = df1.select(check($"speech").as("speech"))
 




    // create new column speechTok with the tokenized speeches using regex
    val df2 = new RegexTokenizer().setInputCol("speech")
      .setOutputCol("speechTok")
      .setMinTokenLength(4)
      .setToLowercase(true)
      .setPattern("[\\s.,!-~'…\"’΄;*^%$@«?|»{}()&–<>/+_ ]")
      .transform(df15)

    // filter all the stopwords from speeches
    val stopwords_gr = ss.sparkContext.textFile(stopwordPath)
                                    .collect
                                    .toSet

    
    // user-defined-function to perform the filtering
    val udf_filter_stop = udf ( (s : mutable.WrappedArray[String]) => s.filter(w => !stopwords_gr.contains(w)))
    
    
 
    // apply the stopwords filter
    val df3 = df2.select(udf_filter_stop($"speechTok").as("rFeatures"))

    // create a count vectorizer model and build a vocabulary based on the speeches column (max words 50000)
    val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("rFeatures")
      .setOutputCol("features")
      .setVocabSize(100000)
      .fit(df3)

    val df4 = cvModel.transform(df3)

    // construct LDA model and fit it on the latest dataframe
    val ldaModel = new LDA().setK(8).setMaxIter(100).fit(df4)

    // print all the topics and the top k words associated with that topic
    val maxWordsPerTopic_ = 10
    ldaModel.describeTopics(maxTermsPerTopic = maxWordsPerTopic_).collect().foreach
    { r => {
        println("Topic: " + r(0))
        val terms:mutable.WrappedArray[Int] = r(1).asInstanceOf[mutable.WrappedArray[Int]]
        terms.foreach {
          t => { println("Term: " + cvModel.vocabulary(t)) }
        }
        println()
      }
    }

  }
}

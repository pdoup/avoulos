import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, CountVectorizerModel, CountVectorizer, IDF, Word2Vec}
import org.apache.spark.ml.linalg.Vector 
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.Window 
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.mllib.feature.Stemmer

object Kmeans {

	final val inputFile = "file:///home/panos/spylons/Greek_Parliament_Proceedings_1989_2020.csv"
	final val stopwords_gr = "file:///home/panos/spylons/stopwords.txt"
	
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
		            .select("member_name", "sitting_date", "speech")
		            .na
		            .drop
		            .withColumn("date_y", to_date($"sitting_date", "dd/MM/yyyy"))
		            .drop("sitting_date")
		        
        val concatSpeechesDF = df.groupBy("member_name")
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

        
        val stemmedDF = new Stemmer().setInputCol("speeches_tok1")
        							.setOutputCol("speeches_tok_stemmed")
        							.setLanguage("Greek")
        							.transform(speechesFilteredDF)

		val cvModel : CountVectorizerModel = new CountVectorizer().setInputCol("speeches_tok_stemmed")
				                                        .setOutputCol("rawFeatures")
				                                        .setMaxDF(0.4f) 
				                                        .setVocabSize(30000)
				                                        .fit(stemmedDF)

	    val cvModelDF = cvModel.transform(stemmedDF)
	    						.drop("speeches", "speeches_tok", "speeches_tok1", "speeches_tok_stemmed")
        

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
        

        val membersDF = concatSpeechesDF.select("member_name")
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
		                            .setMaxIter(10)
		                            .fit(word2vecDF)
		                            .transform(word2vecDF)
		                            .cache()
		
		println("=============================================")
		println("Evaluating the clustering w/ silhouette score...")
		

		val silhouette = new ClusteringEvaluator().setFeaturesCol("embeddings").evaluate(kmeansDF)
		println(f"score : $silhouette%.7f")
		
		println("=============================================")

	    kmeansDF.groupBy("prediction").count().show(20, false)
		
		kmeansDF.select("name", "prediction").write.option("header", true).csv(s"kmeans_member_K_${k}_stemming")
		
		println("Output written to csv file")

        ss.stop()
	}
}
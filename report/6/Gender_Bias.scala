/* 
   NOTE : During our experiments we run this as a jupyter notebook, so it lacks the normal structure of a scala file (main methods etc.)
*/

// Our goal here is to find how women are represented in the parliament in terms of the positions they hold (υπουργοί, υφυπουργοί, βουλευτές)
// and how these roles change over the years. We chose a window of 4 years to observe this change and furthermore we created some plots to
// inspect the results visually


import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, RegexTokenizer}
import org.apache.spark.sql.functions.udf
import scala.collection.mutable
import org.apache.spark.sql.functions._

val df = spark.read.option("header", true)
            .csv("file:///home/ozzy/Desktop/bd/dtst.csv")
            .select("member_name","sitting_date", "roles","member_gender")
            .na
            .drop

// keep rows of a specific year
import org.apache.spark.sql.functions.{to_date, to_timestamp}

val df_date = df.withColumn("date_y", to_date($"sitting_date", "dd/MM/yyyy")).drop("sitting_date")

val ss = SparkSession.builder().appName("sixth").master("local[*]").getOrCreate()
    ss.sparkContext.setLogLevel("ERROR")

for ( year <- 1990 to 2020 by 4){

    var df_years = df_date.where(s"year(date_y) == ${year}")
    var women = df_years.select("member_name", "member_gender").dropDuplicates().groupBy("member_gender").count
    var count = df_years.select("member_name", "member_gender").dropDuplicates().count
    println(year)
    val women1 = women.withColumn("No of women in Vouli %", round(women("count")/count,3)*100)


    var roles = df_years.select("member_name", "member_gender", "roles").dropDuplicates()

    var temp = roles.withColumn("roles1", regexp_replace(roles.col("roles"), "[\\[\\]'']", "")).drop("roles")

    var nonvoul = temp.where(s"roles1 != 'βουλευτης'").groupBy("member_gender").count
    var sumSteps1 =  nonvoul.agg(sum("count")).first.get(0)
    var nonvoul1 = nonvoul.withColumn("Other than Vouleutis %", round(nonvoul("count")/sumSteps1,3)*100)

    var upourgos = temp.where(s"roles1 like 'υπουργος%'").groupBy("member_gender").count
    var sumSteps2 =  upourgos.agg(sum("count")).first.get(0)
    var upourgos1 =upourgos.withColumn("Ypourgos %", round(upourgos("count")/sumSteps2,3)*100)

    var ufipourgos = temp.where(s"roles1 like 'υφυπουργος%'").groupBy("member_gender").count
    var sumSteps3 =  ufipourgos.agg(sum("count")).first.get(0)
    var ufipourgos1 =ufipourgos.withColumn("Yfipourgos %", round(ufipourgos("count")/sumSteps3,3)*100)
    nonvoul1.join(women, "member_gender").join(women1, "member_gender").join(upourgos1,"member_gender").join(ufipourgos1,"member_gender").drop("count").show

}

var df_years = df_date.where(s"year(date_y) == 2020")
var women = df_years.select("member_name", "member_gender").dropDuplicates().groupBy("member_gender").count
var count = df_years.select("member_name", "member_gender").dropDuplicates().count
var women1 = women.withColumn("percentage %", round(women("count")/count,3)*100)

var roles = df_years.select("member_name", "member_gender", "roles").dropDuplicates()

var temp = roles.withColumn("roles1", regexp_replace(roles.col("roles"), "[\\[\\]'']", "")).drop("roles")

var nonvoul = temp.where(s"roles1 != 'βουλευτης'").groupBy("member_gender").count
var sumSteps1 =  nonvoul.agg(sum("count")).first.get(0)
var nonvoul1 = nonvoul.withColumn("Other than Vouleutis %", round(nonvoul("count")/sumSteps1,3)*100)

var upourgos = temp.where(s"roles1 like 'υπουργος%'").groupBy("member_gender").count
var sumSteps2 =  upourgos.agg(sum("count")).first.get(0)
var upourgos1 =upourgos.withColumn("Ypourgos %", round(upourgos("count")/sumSteps2,3)*100)

var ufipourgos = temp.where(s"roles1 like 'υφυπουργος%'").groupBy("member_gender").count
var sumSteps3 =  ufipourgos.agg(sum("count")).first.get(0)
var ufipourgos1 =ufipourgos.withColumn("Yfipourgos %", round(ufipourgos("count")/sumSteps3,3)*100)

nonvoul1.join(women, "member_gender").join(women1, "member_gender").join(upourgos1,"member_gender").join(ufipourgos1,"member_gender").drop("count").show










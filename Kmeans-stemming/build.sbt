name := "Kmeans"

version := "0.0.1"

scalaVersion := "2.11.12"

val sparkVersion = "2.4.8"

libraryDependencies ++= Seq(
	"org.apache.spark" %% "spark-core" % sparkVersion,
	"org.apache.spark" %% "spark-mllib" % sparkVersion,
	"com.github.master" % "spark-stemming_2.10" % "0.2.1"
)
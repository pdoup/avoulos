name := "Kmeans"

version := "0.0.1"

scalaVersion := "2.12.8"

val sparkVersion = "3.0.1"

libraryDependencies ++= Seq(
	"org.apache.spark" %% "spark-core" % sparkVersion,
	"org.apache.spark" %% "spark-mllib" % sparkVersion,
	"org.apache.commons" % "commons-lang3" % "3.12.0"
)

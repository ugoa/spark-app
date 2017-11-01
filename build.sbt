import Dependencies._

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "com.qubida",
      scalaVersion := "2.11.7",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "spark-livy",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "2.1.0",
      "org.apache.spark" %% "spark-sql" % "2.1.0",
      "org.apache.spark" %% "spark-hive" % "2.1.0",
      "org.apache.spark" %% "spark-streaming" % "2.1.0",
      "org.elasticsearch" %% "elasticsearch-spark" % "2.4.4",
      "org.apache.spark" %% "spark-mllib" % "2.1.0",
      "com.cloudera.livy" %% "livy-core" % "0.3.0",
      "com.cloudera.livy" %% "livy-scala-api" % "0.3.0",
      "com.cloudera.livy" % "livy-api" % "0.3.0",
      "com.cloudera.livy" % "livy-client-http" % "0.3.0"
    ),
    libraryDependencies += scalaTest % Test
  )

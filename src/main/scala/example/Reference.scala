package example

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.sql.types.{StringType, StructField, StructType}


import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

class Reference {

  val conf = new SparkConf().setAppName("ODataToElasticSparkWriter")
  val sc = new SparkContext(conf)
  val spark = new SQLContext(sc).sparkSession

  // Load and parse the data file, converting it to a DataFrame.
  val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

  // Index labels, adding metadata to the label column.
  // Fit on whole dataset to include all labels in index.
  val labelIndexer = new StringIndexer()
    .setInputCol("label")
    .setOutputCol("indexedLabel")
    .fit(data)


  labelIndexer.write.overwrite()
  labelIndexer.getInputCol

  // Automatically identify categorical features, and index them.
  // Set maxCategories so features with > 4 distinct values are treated as continuous.
  val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(4)
    .fit(data)

  val encoder = new OneHotEncoder().setDropLast(true)


  import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
  val rfregresser = new RandomForestRegressor()
    .setLabelCol("label")
    .setFeaturesCol("indexedFeatures")
    .setImpurity("asdf")


  val si = new StringIndexer()  // Split the data into training and test sets (30% held out for testing).
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

  val tokenizer = new Tokenizer().setInputCol("asdf").setOutputCol("asdf")
  val regexTokenizer = new RegexTokenizer().setInputCol("asdf").setOutputCol("asdf").setGaps(true).setMinTokenLength(4)

  val cv = new CountVectorizer().setInputCol("asdf").setOutputCol("asdf").setVocabSize(3).setMinDF(2.4)

  // Train a RandomForest model.
  val rf = new RandomForestClassifier()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("indexedFeatures")
    .setNumTrees(10)

  rf.featureSubsetStrategy

  val linearReg = new LinearRegression()
    .setMaxIter(10)
    .setElasticNetParam(23)
    .setFitIntercept(false)
    .setRegParam(10)
    .setTol(23)
    .setStandardization(true)
    .setWeightCol("")
    .setSolver("auto")
    .setLabelCol("asdf")
    .setFeaturesCol("sdf")
    .setPredictionCol("sdf")


  val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setWeightCol("s")
  val pa = lr.params.head
  lr.set(pa, "s")
  lr.getWeightCol
  lr.fitIntercept
  import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
  val paramGrid = new ParamGridBuilder()
    .addGrid(lr.regParam, Array(0.1, 0.01))
    .addGrid(lr.regParam, Array(1, 5))
    .build()


  val colParamsToBeFormatted = Array(
    ("labelCol", "getLabelCol", "setLabelCol"),
    ("featuresCol", "getFeaturesCol", "setFeaturesCol"),
    ("rawPredictionCol", "getRawPredictionCol", "setRawPredictionCol"),
    ("weightCol", "getWeightCol", "getWeightCol")
  )

  val methods = lr.getClass.getMethods


  lr.setFitIntercept(false).setMaxIter(29303).setRegParam(2.5).setTol(2.8)
    .setStandardization(true).setWeightCol("ds").setLabelCol("sdf").setThreshold(23.3)

  val binaryEval = new BinaryClassificationEvaluator().setMetricName("adf").setRawPredictionCol("sd").setLabelCol("sdf")
  binaryEval.getRawPredictionCol

  val p = binaryEval.params.head
  binaryEval.set(p, "ds")


  val remover = new StopWordsRemover().setCaseSensitive(true)

  import org.apache.spark.ml.evaluation.RegressionEvaluator

  val re = new RegressionEvaluator().setLabelCol("").setMetricName("lz").setPredictionCol("ree")

  // Convert indexed labels back to original labels.
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)

  // Chain indexers and forest in a Pipeline.
  val pipeline = new Pipeline()
    .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter, tokenizer))

  val cvalidatior = new CrossValidator().setEvaluator(re).setNumFolds(3).setEstimator(lr)

  // Train model. This also runs the indexers.
  val model = pipeline.fit(trainingData)

  import org.apache.spark.sql.functions.col
  data.join(data, col("df1.te") === col("df2.tf") && col("asdf") == col("sdfs"), "inner")

  // Make predictions.
  val predictions = model.transform(testData)

  // Select example rows to display.
  predictions.select("predictedLabel", "label", "features").show(5)

  // Select (prediction, true label) and compute test error.
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println("Test Error = " + (1.0 - accuracy))

  val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
  println("Learned classification forest model:\n" + rfModel.toDebugString)


  import org.apache.spark.sql.{DataFrame, Dataset}
  import org.apache.spark.sql.types.{StructField, StructType}
  import org.apache.spark.ml.Transformer
  import org.apache.spark.ml.util.Identifiable
  import org.apache.spark.ml.param.Param
  import org.apache.spark.ml.param.ParamMap
  class ReplaceColumn(override val uid: String) extends Transformer {

    final val originCol: Param[String] = new Param[String](uid, "originCol", "input column name")
    final val newCol: Param[String] = new Param[String](uid, "newCol", "input column name")

    final def getOriginCol: String = $(originCol)
    final def getNewCol: String = $(newCol)

    final def setOriginCol(value: String): this.type = set(originCol, value)
    final def setNewCol(value: String): this.type = set(newCol, value)

    def this() = this(Identifiable.randomUID("replaceColumn"))

    override def transform(dataset: Dataset[_]): DataFrame = {
      val orderedFields = dataset.schema.fieldNames.filterNot(_ == getNewCol)
      val fieldsWithoutOrigin = dataset.schema.fieldNames.filterNot(_ == getOriginCol)

      dataset
        .select(fieldsWithoutOrigin.head, fieldsWithoutOrigin.tail: _*)
        .withColumnRenamed(getNewCol, getOriginCol)
        .select(orderedFields.head, orderedFields.tail: _*)
    }

    override def copy(extra: ParamMap): ReplaceColumn = defaultCopy(extra)

    override def transformSchema(schema: StructType): StructType = {
      val newField = schema.find(_.name == getNewCol).get
      val fields = schema.map { structField =>
        if (structField.name == getOriginCol) {
          StructField(getOriginCol, newField.dataType, newField.nullable, newField.metadata)
        } else structField
      }
      StructType(fields.filterNot(_.name == getNewCol))
    }
  }
}

class GridSearch {
  import org.apache.spark.ml.Pipeline
  import org.apache.spark.ml.classification.LogisticRegression
  import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
  import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
  import org.apache.spark.ml.linalg.Vector
  import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

  import org.apache.spark.sql.Row

  val output5_1_Estimator = new org.apache.spark.ml.classification.RandomForestClassifier().setMaxDepth(5).setMaxBins(32).setMinInstancesPerNode(1).setMinInfoGain(0).setMaxMemoryInMB(256).setCacheNodeIds(true).setCheckpointInterval(10).setImpurity("gini").setSubsamplingRate(1).setSeed(0).setFeatureSubsetStrategy("auto").setNumTrees(20).setLabelCol("default_payment_next_month").setFeaturesCol("features").setProbabilityCol("probability").setRawPredictionCol("rawPrediction").setPredictionCol("prediction").setSeed(2)



  val conf = new SparkConf().setAppName("ODataToElasticSparkWriter")
  val sc = new SparkContext(conf)
  val spark = new SQLContext(sc).sparkSession

  // Load and parse the data file, converting it to a DataFrame.
  val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

  // Prepare training data from a list of (id, text, label) tuples.
  val training = spark.createDataFrame(Seq(
    (0L, "a b c d e spark", 1.0),
    (1L, "b d", 0.0),
    (2L, "spark f g h", 1.0),
    (3L, "hadoop mapreduce", 0.0),
    (4L, "b spark who", 1.0),
    (5L, "g d a y", 0.0),
    (6L, "spark fly", 1.0),
    (7L, "was mapreduce", 0.0),
    (8L, "e spark program", 1.0),
    (9L, "a e c l", 0.0),
    (10L, "spark compile", 1.0),
    (11L, "hadoop software", 0.0)
  )).toDF("id", "text", "label")

  // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
  val tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("words")
  val hashingTF = new HashingTF()
    .setInputCol(tokenizer.getOutputCol)
    .setOutputCol("features")
  val lr = new LogisticRegression()
    .setMaxIter(10)
  val pipeline = new Pipeline()
    .setStages(Array(tokenizer, hashingTF, lr))


  // We use a ParamGridBuilder to construct a grid of parameters to search over.
  // With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
  // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
  val paramGrid: Array[ParamMap] = new ParamGridBuilder()
    .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
    .addGrid(lr.regParam, Array(0.1, 0.01))
    .addGrid(lr.fitIntercept)
    .addGrid(output5_1_Estimator.seed, Array(0.toLong))
    .addGrid(output5_1_Estimator.impurity, Array("e"))
    .build()


  // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
  // This will allow us to jointly choose parameters for all Pipeline stages.
  // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
  // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
  // is areaUnderROC.
  val bce = new BinaryClassificationEvaluator
  val cv = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(bce)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(2)  // Use 3+ in practice

  // Run cross-validation, and choose the best set of parameters.
  val cvModel = cv.fit(training)
  paramGrid.head.toSeq
  cvModel.avgMetrics

  val headers = paramGrid.head.toSeq.map(_.param.name).sorted :+ "metric"

  val zippedData = paramGrid.zip(cvModel.avgMetrics)
  val sortedData = if (bce.isLargerBetter) zippedData.sortBy(_._2).reverse else zippedData.sortBy(_._2)
  val rows = sortedData.map { case (pg, metric) =>
    val values = pg.toSeq.sortBy(_.param.name).map(_.value.toString) :+ metric.toString
    Row(values: _*)
  }
  val rdd = sc.parallelize(rows)
  import org.apache.spark.sql.types.{StringType, StructField, StructType}
  val structFields = headers.map(StructField(_, StringType, true)).toList
  val schema = StructType(structFields)
  val df = spark.createDataFrame(rdd, schema)
}


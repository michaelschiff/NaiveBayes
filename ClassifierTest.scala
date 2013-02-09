import java.io.File
import scala.io.Source
import NaiveBayes._
import ngrams.tokenFunctions._
package ClassifierTest {
  object ClassifierTest extends App {
    var posExamples: List[Array[String]] = List()
    var negExamples: List[Array[String]] = List()

    println("Gathering Training Documents")
    //iterate through positive examples
    for ( file <- new File("review_polarity/txt_sentoken/pos").listFiles.toIterator if file.isFile ) {
      var tokens: Array[String] = Array()
      for (line <- Source fromFile file getLines ) {
        var splitLine = line.split(" ")
        tokens = tokens ++ splitLine
      }
      //tokens = ngrams(tokens, 2)
      posExamples = tokens :: posExamples
    }
    
    //iterate through negative examples
    for ( file <- new File("review_polarity/txt_sentoken/neg").listFiles.toIterator if file.isFile ) {
      var tokens: Array[String] = Array()
      for (line <- Source fromFile file getLines ) {
        var splitLine = line.split(" ")
        tokens = tokens ++ splitLine
      }
      //tokens = ngrams(tokens, 2)
      negExamples = tokens :: negExamples
    }
    
    //Build a map of the training data
    val trainingData: Map[Int, List[Array[String]]] = Map(0 -> posExamples, 1 -> negExamples)

    val classifier: BernoulliNB = new BernoulliNB(trainingData)

    
    println("Testing Classifier")
    println("  |  Gathering Test Documents")
    var tests = Map[Array[String], Int]()
    for ( file <- new File("review_polarity/txt_sentoken/pos_test").listFiles.toIterator if file.isFile ) {
      var tokens: Array[String] = Array()
      for (line <- Source fromFile file getLines ) {
        var splitLine = line.split(" ")
        tokens = tokens ++ splitLine
      }
      //tokens = ngrams(tokens, 2)
      tests += (tokens -> 0)
    }
    for ( file <- new File("review_polarity/txt_sentoken/neg_test").listFiles.toIterator if file.isFile ) {
      var tokens: Array[String] = Array()
      for (line <- Source fromFile file getLines ) {
        var splitLine = line.split(" ")
        tokens = tokens ++ splitLine
      }
      //tokens = ngrams(tokens, 2)
      tests += (tokens -> 1)
    }
    println("  |  Classifying documents")
    var tp = 0.0
    var fp = 0.0
    var tn = 0.0
    var fn = 0.0
    var tp1 = 0.0
    var fp1 = 0.0
    var tn1 = 0.0
    var fn1 = 0.0
    for ( test <- tests ) {
      val prediction = classifier.predict(test._1)
      val label = test._2
      if (prediction == 0 && label == 0) {
        tp += 1
        tn1 += 1
      } else if (prediction == 0 && label == 1) {
        fp += 1
        fn1 += 1
      } else if (prediction == 1 && label == 0) {
        fn += 1
        fp1 += 1
      } else if (prediction == 1 && label == 1) {
        tn += 1
        tp1 += 1
      }
    }
    val red = "\033[1;31m"
    val green = "\033[1;32m"
    val blue = "\033[1;34m"
    val default = "\033[0m"
    def color(str: String, c: String): String = c+str+default

    println("  |  Positive Scores:")
    println("  |    "+color("Precision: ", red) + (tp / (tp + fp)))
    println("  |    "+color("Recall: ", red) + (tp / (tp + fn)))
    println("  |    "+color("Accuracy: ", red) + (100*((tp+tn) / (tp+fp+tn+fn))) + "%")
    println("  |  Negative Scores:")
    println("  |    "+color("Precision: ", red) + (tp1 / (tp1 + fp1)))
    println("  |    "+color("Recall: ", red) + (tp1 / (tp1 + fn1)))
    println("  |    "+color("Accuracy: ", red) + (100*((tp1+tn1) / (tp1 + fp1 + tn1 + fn1))) + "%")
  } //closes object
} //closes package

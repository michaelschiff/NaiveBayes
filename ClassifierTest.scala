import java.io.File
import scala.io.Source
import NaiveBayes._
package ClassifierTest {
  object ClassifierTest extends App {
  
    var posExamples: List[Array[String]] = List()
    var negExamples: List[Array[String]] = List()

    //iterate through positive examples
    for ( file <- new File("review_polarity/txt_sentoken/pos").listFiles.toIterator if file.isFile ) {
      var tokens: Array[String] = Array()
      for (line <- Source fromFile file getLines ) {
        val splitLine = line.split(" ")
        tokens = tokens ++ splitLine
      }
      posExamples = tokens :: posExamples
    }
    
    //iterate through negative examples
    for ( file <- new File("review_polarity/txt_sentoken/neg").listFiles.toIterator if file.isFile ) {
      var tokens: Array[String] = Array()
      for (line <- Source fromFile file getLines ) {
        val splitLine = line.split(" ")
        tokens = tokens ++ splitLine
      }
      negExamples = tokens :: negExamples
    }
    
    //Build a map of the training data
    val trainingData: Map[Int, List[Array[String]]] = Map(0 -> posExamples, 1 -> negExamples)

    val classifier: BernoulliNB = new BernoulliNB(trainingData)

    //test on positive docs
    var posScore = 0
    var posOutOf = 0
    for ( file <- new File("review_polarity/txt_sentoken/pos_test").listFiles.toIterator if file.isFile ) {
      var tokens: Array[String] = Array()
      for (line <- Source fromFile file getLines ) {
        val splitLine = line.split(" ")
        tokens = tokens ++ splitLine
      }
      val result = classifier.predict(tokens)
      if (result == 0) { posScore += 1 }
      posOutOf += 1
    }
    
    //test on negative docs
    var negScore = 0
    var negOutOf = 0
    for ( file <- new File("review_polarity/txt_sentoken/neg_test").listFiles.toIterator if file.isFile ) {
      var tokens: Array[String] = Array()
      for (line <- Source fromFile file getLines ) {
        val splitLine = line.split(" ")
        tokens = tokens ++ splitLine
      }
      val result = classifier.predict(tokens)
      if (result == 1) { negScore += 1 }
      negOutOf += 1
    }
    
    //Print out the scores
    println("Correctly classified " + posScore + " out of " + posOutOf + " positive test documents")
    println("Correctly classified " + negScore + " out of " + negOutOf + " negative test documents")

  }
}

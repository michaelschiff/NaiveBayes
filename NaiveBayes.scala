import BIDMat.{Mat, FMat, DMat, IMat, CMat, BMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat, HMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
package NaiveBayes {
  class BernoulliNB(trainingData: Map[Int, List[Array[String]]]) {
    println("Initializing and Training Classifier")
    Mat.noMKL=true  //can I comment this out for better performance???
    flip;
    ///////////////Count the number of training documents////////////////////// 
    private var numDocs: Float = 0
    for ( docSet <- trainingData ) {
      numDocs += docSet._2.length
    }
    println("  |  Counted " + numDocs + " training documents.")
    /////////////////////////////////////////////////////////////////////////////

    ///////////////////Construct the vocabulary of words////////////////////////
    private var vocabulary: Set[String] = Set()
    for ( docSet <- trainingData ) {
      for ( document <- docSet._2 ) {
        for ( word <- document) {
          vocabulary += word
        }
      }
    }
    private var vocabularyMap: Map[String, Int] = Map[String,Int]()
    private var revVocabularyMap: Map[Int, String] = Map[Int, String]()
    for ( (term,index) <- vocabulary.zipWithIndex ) { 
      vocabularyMap += (term -> index) 
      revVocabularyMap += (index -> term)
    } 
    println("  |  Built vocabulary of " + vocabulary.size + " tokens.")
    //////////////////////////////////////////////////////////////////////////////
    
    /////////////////////Construct the prior probabilities////////////////////////
    private var priors = zeros(1, trainingData.keySet.size) //1 row, labelsCols
    for ( docSet <- trainingData ) {
      priors(0, docSet._1) = docSet._2.length / numDocs
    }
    priors = ln(priors)
    println("  |  Calculated prior probabilities")
    //////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////
    ////Initialize and fill with dense matrix and convert before math
    //var matrixes: Map[Int, FMat] = Map()
    private var matrixes = new Array[FMat](trainingData.keys.size)
    for ( docSet <- trainingData ) {
      var mat = FMat(vocabulary.size, docSet._2.length)
      for ( (document, docIndex) <- docSet._2.zipWithIndex ) {
        for ( word <- document ) {
          mat(vocabularyMap(word).asInstanceOf[Int], docIndex) = 1 
        }
      }
      var wordFreqs = sum(mat, 2) // mat is now the column vector. each row holds the number of docs containg that word
      wordFreqs += ones(wordFreqs.size, 1) //add the column vector of ones for smoothing
      //divide every elt by the number docs in class + 2
      wordFreqs = wordFreqs/@(docSet._2.length+2)
      matrixes(docSet._1) = wordFreqs
    }
    //JOIN THE INDIVIDUAL CONDPROBS into one matrix.
    private var condProbs = matrixes.reduceLeft((col1,col2)=>col1\col2)
    private var inverseCondProbs = ln(ones(vocabulary.size, 1) - condProbs)
    condProbs = ln(condProbs)
    println("  |  Built conditional probability table")
    
    //////////////////////////////////////////////////////////////////////////////
    println("Training Speed: " + flop)

    //////////////////////Predictions////////////////////////////////////////////
    def predict(example: Array[String]): Int = {
      flip;
      val a: FMat = zeros(1, vocabulary.size)
      val b: FMat = ones(1, vocabulary.size)
      for (word <- example if vocabulary contains word ) {
        a(0, vocabularyMap(word).asInstanceOf[Int]) = 1
        b(0, vocabularyMap(word).asInstanceOf[Int]) = 0
      }
      val prob = a * condProbs
      val inverseProb = b * inverseCondProbs
      val result = priors + prob + inverseProb
        
      println(flop)
      //This is a hack.  find a more general way to do the argmax. This one relies on their being only 2 labels
      if (result(0,0) > result(0,1)) { return 0 }
      else { return 1 }
    }

    def analyzeTokens(): Unit = {
      //this function outputs the top five tokens that best distinguish a positive doc and the top five tokens that best distinguis a negative doc.  Assumes only two labels
      var posImpacts: FMat = condProbs(?, 0) - condProbs(?, 1)
      var negImpacts: FMat = condProbs(?, 1) - condProbs(?, 0)
      println("  |  Words Most Indicative of a Positive Review:")
      var i = 0
      while (i < 5) {
        var maxPos = maxi(posImpacts, 1)(0,0) 
        for (x <- (0 to posImpacts.size-1)) { 
          if (posImpacts(x, 0) == maxPos) { 
            posImpacts(x, 0) = 0
            println("  |    "+revVocabularyMap(x) + ": " + maxPos)
            i += 1
          }
        }
      }
      println("  |  Words Most Indicative of a Negative Review:")
      i = 0
      while (i < 5) {
        var maxNeg = maxi(negImpacts, 1)(0,0) 
        for (x <- (0 to negImpacts.size-1)) { 
          if (negImpacts(x, 0) == maxNeg) { 
            negImpacts(x, 0) = 0
            println("  |    "+revVocabularyMap(x) + ": " + maxNeg)
            i += 1
          }
        }
      }
    }
  }


}

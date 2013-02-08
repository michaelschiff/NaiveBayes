import BIDMat.{Mat, FMat, DMat, IMat, CMat, BMat, CSMat, SMat, SDMat, GMat, GIMat, GSMat, HMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
package NaiveBayes {
  class BernoulliNB(trainingData: Map[Int, List[Array[String]]]) {
    println("Initializing and Training Classifier")
    Mat.noMKL=true  //can I comment this out for better performance???
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
    for ( (term,index) <- vocabulary.zipWithIndex ) { vocabularyMap += (term -> index) } 
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
  
    //////////////////////Predictions////////////////////////////////////////////
    def predict(example: Array[String]): Int = {
      val a: FMat = zeros(1, vocabulary.size)
      val b: FMat = ones(1, vocabulary.size)
      for (word <- example if vocabulary contains word ) {
        a(0, vocabularyMap(word).asInstanceOf[Int]) = 1
        b(0, vocabularyMap(word).asInstanceOf[Int]) = 0
      }
      val prob = a * condProbs
      val inverseProb = b * inverseCondProbs
      val result = priors + prob + inverseProb
      //println(result) ## Im getting negative probabilities. Why?
      
      //This is a hack.  find a more general way to do the argmax.
      if (result(0,0) > result(0,1)) { return 0 }
      else { return 1 }
    }
  }
}

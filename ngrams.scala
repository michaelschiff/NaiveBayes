package ngrams
object tokenFunctions {
  def ngrams(wordArr: Array[String], n: Int): Array[String] = {
    var toReturn = Array[String]()
    for (i <- (1 to n)) {
      for ( x <- wordArr.sliding(i) ) {
        toReturn = Array(x.reduceLeft((j,k) => j.mkString+" "+k.mkString)) ++ toReturn
      }
    }
    return toReturn
  }
}

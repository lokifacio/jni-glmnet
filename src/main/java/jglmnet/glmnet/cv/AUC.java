package jglmnet.glmnet.cv;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

import java.util.*;

/**
 * @author Jorge Peña
 */
public class AUC {

  public static void mat(DoubleMatrix1D y, DenseDoubleMatrix2D prob, DoubleMatrix1D weights)
  {
    DoubleMatrix1D Weights = weights.copy();
    Weights.assign(y, (a, b) -> a*b);
    int ny = (int)y.size();

    List<Integer> Y = Collections.nCopies(ny, 0);
    Y.addAll(Collections.nCopies(ny, 1));

//    List<Double> Prob = new ArrayList(Arrays.asList(prob.toArray()));
//    Prob.addAll(prob.toArray());

    //auc(Y, Prob, Weights)
  }

  private static void auc(){
//    if (missing(w)) {
//      rprob = rank(prob)
//      n1 = sum(y)
//      n0 = length(y) - n1
//      u = sum(rprob[y == 1]) - n1 * (n1 + 1)/2
//      u/(n1 * n0)
//    }
//    else {

//    rprob = runif(length(prob))
//      op = order(prob, rprob)
//      y = y[op]
//      w = w[op]
//      cw = cumsum(w)
//      w1 = w[y == 1]
//      cw1 = cumsum(w1)
//      wauc = sum(w1 * (cw[y == 1] - cw1))
//      sumw = cw1[length(cw1)]
//      sumw = sumw * (cw[length(cw)] - sumw)
//      wauc/sumw
//    }
  }
}

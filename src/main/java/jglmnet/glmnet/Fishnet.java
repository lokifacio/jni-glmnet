package jglmnet.glmnet;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseColumnDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;

import java.util.*;

/**
 * @author Jorge Peña
 */
class Fishnet {

  private static class ClassificationModel extends jglmnet.glmnet.ClassificationModel {

    ClassificationModel(double intercept, DoubleMatrix1D betas, double lambda, boolean hasOffset) {
      super(intercept, betas, lambda, hasOffset);
    }

    @Override
    public double invLink(double x) {
      return Math.exp(x);
    }
  }

  private static class ClassificationModelSet extends jglmnet.glmnet.ClassificationModelSet {

    ClassificationModelSet(int numPasses, int numFits, double[] intercepts, double[] coeffs, int[] coeffPtrs, int[] coeffCnts, double[] lambdas, int columns, int maxPathFeatures, boolean hasOffset) {
      super(numPasses, numFits, intercepts, coeffs, coeffPtrs, coeffCnts, lambdas, columns, maxPathFeatures, hasOffset);
    }

    @Override
    jglmnet.glmnet.ClassificationModel createModel(double intercept, DoubleMatrix1D betas, double lambda) {
      return new Fishnet.ClassificationModel(intercept, betas, lambda, hasOffset());
    }
  }

  static ClassificationModelSet fit
      ( DoubleMatrix2D x
      , DoubleMatrix1D y
      , DoubleMatrix1D weights
      , DoubleMatrix1D offset
      , double alpha
      , DoubleMatrix1D jd
      , DenseDoubleMatrix1D vp
      , DenseColumnDoubleMatrix2D cl
      , int ne // maxFinalFeatures
      , int nx // maxPathFeatures
      , int nlam
      , double flmin
      , double[] ulam
      , double thresh
      , int isd
      , int intr
      , String[] vnames
      , int maxit
      )
      throws Exception {

    int nobs  = x.rows();
    int nvars = x.columns();

    if (y.size() != nobs) {
      throw new Exception("x and y have different number of rows in call to glmnet");
    }

    Map<Double, Long> count = Classifiers.getClassCount(y);

    // nc = number of classes (distinct outcome values)
    int nc = count.size();

    for (int i = 0; i < y.size(); ++i) {
      if (y.get(i) < 0) {
        throw new Exception("Negative responses encountered; not permited for Poisson family");
      }
    }

    DoubleMatrix1D w = weights;
    if(w == null) {
      w = y.copy().assign(1);
    }

    boolean isOffset = offset != null;
    DenseDoubleMatrix1D o;
    if (isOffset) {
      o = new DenseDoubleMatrix1D(offset.toArray());
    } else {
      o = new DenseDoubleMatrix1D(nobs);
      o.assign(0);
    }

    DenseDoubleMatrix1D dy = new DenseDoubleMatrix1D(y.toArray());

    // Check for size limitations
    int maxVars = Integer.MAX_VALUE/(nlam*nc);
    if(nx > maxVars) {
      throw new Exception("Integer overflow; num_classes*num_lambda*pmax should not exceed Integer.MAX_VALUE. Reduce pmax to be below " + maxVars);
    }

    int err = 0;

    ClassificationModelSet fit = null;
    boolean isSparse = false;
    if(isSparse) {
      // spfishnet
    } else {
      DenseColumnDoubleMatrix2D dcx;
      if (x instanceof DenseColumnDoubleMatrix2D) {
         dcx = (DenseColumnDoubleMatrix2D) x;
      } else {
        dcx = new DenseColumnDoubleMatrix2D(x.rows(), x.columns());
        dcx.assign(x);
      }

      double[] outIntercepts = new double[nlam];
      double[] outCoeffs = new double[nx * nlam];
      int[] outCoeffPtrs = new int[nx];
      int[] outCoeffCnts = new int[nlam];
      double[] outDev0 = new double[nlam];
      double[] outFdev = new double[nlam];
      double[] outLambdas = new double[nlam];
      int[] outNumPasses = new int[1];
      int[] outNumFits = new int[1];

      Arrays.fill(outLambdas, Double.NaN);

      err = Fortran.fishnet(
          alpha,
          dy.elements(),
          dcx.elements(),
          o.elements(),
          w.toArray(),
          new int[1],
          vp.elements(),
          cl.elements(),
          ne,
          nx,
          nlam,
          flmin,
          ulam,
          thresh,
          isd,
          intr,
          maxit,
          outNumFits,
          outIntercepts,
          outCoeffs,
          outCoeffPtrs,
          outCoeffCnts,
          outDev0,
          outFdev,
          outLambdas,
          outNumPasses);

      if (err != 0 ) {
        throw new Exception("glmnet error: " + err);
      }

      fit = new ClassificationModelSet(outNumPasses[0], outNumFits[0], outIntercepts, outCoeffs, outCoeffPtrs, outCoeffCnts, outLambdas, nvars, nx, isOffset);
    }
    return fit;
  }
}

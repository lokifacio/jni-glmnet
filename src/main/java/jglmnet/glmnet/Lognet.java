package jglmnet.glmnet;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseColumnDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;

import java.util.*;
import java.util.stream.Stream;

/**
 * @author Jorge Peña
 */
class Lognet {

  private static class ClassificationModel extends jglmnet.glmnet.ClassificationModel {

    ClassificationModel(double intercept, DoubleMatrix1D betas, double lambda, boolean hasOffset) {
      super(intercept, betas, lambda, hasOffset);
    }

    @Override
    public double invLink(double x) {
      return logit(x);
    }

    private double logit(double x) {
      return 1.0 / (1.0 + Math.exp(-x));
    }
  }

  private static class ClassificationModelSet extends jglmnet.glmnet.ClassificationModelSet {

    ClassificationModelSet(int numPasses, int numFits, double[] intercepts, double[] coeffs, int[] coeffPtrs, int[] coeffCnts, double[] lambdas, int columns, int maxPathFeatures, boolean hasOffset) {
      super(numPasses, numFits, intercepts, coeffs, coeffPtrs, coeffCnts, lambdas, columns, maxPathFeatures, hasOffset);
    }

    @Override
    jglmnet.glmnet.ClassificationModel createModel(double intercept, DoubleMatrix1D betas, double lambda) {
      return new ClassificationModel(intercept, betas, lambda, hasOffset());
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
      , int kopt
      , Family family
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

    long minClass = count.values().stream().min(Long::compare).get();

    if(minClass <= 1) {
      throw new Exception("one multinomial or binomial class has 1 or 0 observations; not allowed");
    } else if(minClass < 8) {
      throw new Exception("one multinomial or binomial class has fewer than 8  observations; dangerous ground");
    }

    List<Double> keys = new ArrayList<>(count.keySet());

    Collections.sort(keys);
    Collections.reverse(keys);

    String[] classNames = Stream.of(keys).map(Object::toString).toArray(String[]::new);

    DoubleMatrix1D w = weights;
    if(w == null) {
      w = y.copy().assign(1);
    }

    DenseColumnDoubleMatrix2D dy = new DenseColumnDoubleMatrix2D(nobs, nc);
    for (int i = 0; i < nobs; ++i) {
      Double c = y.get(i);
      dy.set(i, keys.indexOf(c), w.get(i));
    }

    // Check for size limitations
    int maxVars = Integer.MAX_VALUE/(nlam*nc);
    if(nx > maxVars) {
      throw new Exception("Integer overflow; num_classes*num_lambda*pmax should not exceed Integer.MAX_VALUE. Reduce pmax to be below " + maxVars);
    }

    // o(no,nc) = observation off-sets for each class
    DoubleMatrix1D o = weights.copy().assign(v -> v>0?1:0);
    if (o.zSum() < o.size()) { // subset the data
      // y=y[o,]
      // if(is.sparse){ # we have to subset this beast with care
      //   x=sparseMatrix(i=jx,p=ix-1,x=x,dims=c(nobs,nvars))[o,,drop=FALSE]
      //   ix=as.integer(x@p+1)
      //   jx=as.integer(x@i+1)
      //   x=as.double(x@x)
      // }
      // else x=x[o,,drop=FALSE]
      // nobs=sum(o)
    } else {
      o = null;
    }

    if(family == Family.Binomial){
      if(nc > 2) {
        throw new Exception("More than two classes; use multinomial family instead in call to glmnet");
      }

      nc = 1; // binomial two-class logistic regression (all output references class 1)
    }

    boolean isOffset = offset != null;

    if(isOffset){
      if (o != null) {
        // o = offset[o,]# we might have zero weights
      } else {
        o = offset;
      }

      if(o.size() != nobs) {
        throw new Exception("offset should have the same number of values as observations in binomial/multinomial call to glmnet");
      }
      // TODO use 2D weight matrix
//      if (o.columns == 1 && nc == 1) {
//        // offset=cbind(offset,-offset)
//      }
//      if(family == Family.Multinomial && o.columns() != nc) {
//        throw new Exception("offset should have same shape as y in multinomial call to glmnet");
//      }
    } else {
      o = new DenseDoubleMatrix1D(nobs);
      o.assign(0);
    }

    int err = 0;

    ClassificationModelSet fit = null;
    boolean isSparse = false;
    if(isSparse) {
      //new Fortran().splognet();
//          parm = alpha, nobs, nvars, nc, x, ix, jx, y, offset, jd, vp, cl, ne = ne, nx, nlam, flmin, ulam, thresh, isd, intr, maxit, kopt,
//          lmu = integer(1),
//          a0 =double(nlam * nc),
//          ca =double(nx * nlam * nc),
//          ia = integer(nx),
//          nin = integer(nlam),
//          nulldev =double(1),
//          dev =double(nlam),
//          alm =double(nlam),
//          nlp = integer(1),
//          jerr = integer(1), PACKAGE = "glmnet"
//                    )
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

      err = Fortran.lognet(
          alpha,
          nc,
          dy.elements(),
          o.toArray(),
          dcx.elements(),
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
          kopt,
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

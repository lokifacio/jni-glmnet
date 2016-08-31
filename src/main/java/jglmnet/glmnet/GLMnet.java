package jglmnet.glmnet;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseColumnDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;

import java.util.Collections;
import java.util.List;

/**
 * Java implementation of R glmnet
 * @author Jorge Peña
 */
public class GLMnet extends GLMnetBase {

  private Integer dfmax;
  private Integer pmax;
  // exlude?
  DoubleMatrix1D penaltyFactor; // rep(1, nvars)
  boolean standardizeResponse = false;

  LogisticType logisticType = LogisticType.Newton;
  MultinomialType multinomialType = MultinomialType.Ungrouped;

  @Override
  public GLMnet setAlpha(double value) {
    super.setAlpha(value);
    return this;
  }

  @Override
  public GLMnet setNLambdas(int value) {
    super.setNLambdas(value);
    return this;
  }

  @Override
  public GLMnet setLambdas(List<Double> lambdas) {
    super.setLambdas(lambdas);
    return this;
  }

  @Override
  public GLMnet setLambdaMinRatio(Double value) {
    super.setLambdaMinRatio(value);
    return this;
  }

  @Override
  public GLMnet setFamily(Family family) {
    super.setFamily(family);
    return this;
  }

  @Override
  public GLMnet setThreshold(double value) {
    super.setThreshold(value);
    return this;
  }

  @Override
  public GLMnet setIntercept(boolean value) {
    super.setIntercept(value);
    return this;
  }

  @Override
  public GLMnet setStandardize(boolean value) {
    super.setStandardize(value);
    return this;
  }

  @Override
  public GLMnet setMaxIter(int n) {
    super.setMaxIter(n);
    return this;
  }

  @Override
  public GLMnet setLimits(List<Double> lower, List<Double> upper) throws Exception {
    super.setLimits(lower, upper);
    return this;
  }

  //        lower.limits = -Inf, upper.limits = Inf, maxit = 1e+05, type.gaussian = ifelse(nvars <
//        500, "covariance", "naive"), type.multinomial = c("ungrouped",
//        "grouped")

  // maximum number of variables allowed to enter largest model
  // (stopping criterion)
  public GLMnetBase setDFMax(int value) {
    dfmax = value;

    return this;
  }

  //nx = maximum number of variables allowed to enter all models
  //     along path (memory allocation, pmax > dfmax).
  public GLMnetBase setPMax(int value) {
    pmax = value;

    return this;
  }

  public GLMnet() {
  }

  public GLMnet(GLMnetBase other) {
    super(other);
  }


  public ClassificationModelSet  fit
      ( DoubleMatrix2D x,
        DoubleMatrix1D y,
        DoubleMatrix1D weights
      ) throws Exception {

    if (x.columns() < 2) {
      throw new Exception("x should be a matrix with 2 or more columns");
    }

    int nobs = x.rows();

    if (weights == null) {
      weights = y.assign(1);
    } else if (weights.size() != y.size()) {

      throw new Exception("Number of elements in weights (" + weights.size() +
          ") not equal to the number of rows of x (" + x.size() + ")");
    }

    int nvars = x.columns();

    if (x.rows() != y.size()) {
      throw new Exception("Number of observations in y (" + y.size() +
          ") not equal to the number of rows of x (" + x.rows() + ")");
    }

//    vnames = colnames(x)
//    if (is.null(vnames))
//    vnames = paste("V", seq(nvars), sep = "")

    int ne = (dfmax == null)?(nvars + 1):dfmax;
    int nx = (pmax == null)?Math.min(ne * 2 + 20, nvars):pmax;

    //TODO: excluir variables
//    if (!missing(exclude)) {
//      jd = match(exclude, seq(nvars), 0)
//      if (!all(jd > 0))
//        stop("Some excluded variables out of range")
//      jd = as.integer(c(length(jd), jd))
//    }
//    else jd = as.integer(0)
    //   jd(jd(1)+1) = predictor variable deletion flag
    //      jd(1) = 0  => use all variables
    //      jd(1) != 0 => do not use variables jd(2)...jd(jd(1)+1)
    DenseDoubleMatrix1D jd = new DenseDoubleMatrix1D(1); // Temporalmente hasta que se haga el codigo de arriba

    penaltyFactor = new DenseDoubleMatrix1D(nvars).assign(1);

    DoubleMatrix1D vp = penaltyFactor;

    //TODO: Parametros internos control glmnet
//        internal.parms = glmnet.control()

    //TODO: Limites
    DenseColumnDoubleMatrix2D cl = new DenseColumnDoubleMatrix2D(2, nvars);

    for (int c = 0; c < nvars; c++) {
      cl.set(0, c, Double.NEGATIVE_INFINITY);
      cl.set(1, c, Double.POSITIVE_INFINITY);
    }
    final int LOWER_LIMIT = 0;
    final int UPPER_LIMIT = 1;

    if (lowerLimits.size() == nvars) {
      for (int c = 0; c < nvars; c++) {
        cl.set(LOWER_LIMIT, c, lowerLimits.get(c));
      }
    } else if (lowerLimits.size() <= 1) {
      final double limit = lowerLimits.isEmpty()?Double.NEGATIVE_INFINITY:lowerLimits.get(0);
      cl.viewRow(LOWER_LIMIT).assign(limit);
    } else {
      throw new Exception("Require length 1 or nvars lower.limits");
    }

    if (upperLimits.size() == nvars) {
      for (int c = 0; c < nvars; c++) {
        cl.set(UPPER_LIMIT, c, upperLimits.get(c));
      }
    } else if (upperLimits.size() <= 1) {
      final double limit = upperLimits.isEmpty()?Double.POSITIVE_INFINITY:upperLimits.get(0);
      cl.viewRow(UPPER_LIMIT).assign(limit);
    } else {
      throw new Exception("Require length 1 or nvars upper.limits");
    }

//    if (any(cl == 0)) {
//      fdev = glmnet.control()$fdev
//      if (fdev != 0) {
//        glmnet.control(fdev = 0)
//        on.exit(glmnet.control(fdev = fdev))
//      }
//    }

    // isd = predictor variable standarization flag:
    //     isd = 0 => regression on original predictor variables
    //     isd = 1 => regression on standardized predictor variables
    //     Note: output solutions always reference original variables locations and scales.
    int isd = standardize?1:0;

    //intr = intercept flag
    //     intr = 0/1 => don't/do include intercept in model
    int intr = intercept?1:0;

    if (!intercept && family == Family.Cox) {
      System.err.println("Warning: Cox model has no intercept");
    }

    //jsd = response variable standardization flag
    //    jsd = 0 => regression using original response variables
    //    jsd = 1 => regression using standardized response variables
    //    Note: output solutions always reference original
    //          variables locations and scales.
    int jsd = standardizeResponse?1:0;

    if (lambdaMinRatio == null) {
      lambdaMinRatio = (nobs < nvars)?0.01:1e-04;
    }

    double   flmin = lambdaMinRatio;
    double[] ulam  = new double[]{0};

    if (lambda == null) {
      if (lambdaMinRatio >= 1) {
        throw new Exception("Lambda.min.ratio should be less than 1");
      }
    }
    else {
      for(Double value : lambda) {
        if (value < 0) {
          throw new Exception("Lambdas should be non-negative");
        }
      }
      Collections.sort(lambda);

      flmin = 1;
      ulam = toArray(lambda);
      nlam = lambda.size();
    }

    boolean isSparse = false;

    //x, ix, jx = predictor matrix in compressed sparse row format
    //TODO: Sparse (en un principio podemos delegarlo a las familias
    //ix = jx = NULL
//    if (inherits(x, "sparseMatrix")) {
//      is.sparse = TRUE
//      x = as(x, "CsparseMatrix")
//      x = as(x, "dgCMatrix")
//      ix = as.integer(x@p + 1)
//      jx = as.integer(x@i + 1)
//      x = as.double(x@x)
//    }

    // kopt = optimization flag
    //     kopt = 0 => Newton-Raphson (recommended)
    //      kpot = 1 => modified Newton-Raphson (sometimes faster)
    //      kpot = 2 => nonzero coefficients same for each class (nc > 1)
    int kopt = -1;
    switch (logisticType) {
      case Newton:
        kopt = 0;
        break;
      case ModifiedNewton:
        kopt = 1;
        break;
    }

    if (family == Family.Multinomial) {
      if (multinomialType == MultinomialType.Grouped) {
        kopt = 2;
      }
    }

    ClassificationModelSet mods = null;

    switch (family) {
      case Gaussian:
        // gaussian = elnet(x, is.sparse, ix, jx,
        // y, weights, offset, type.gaussian, alpha, nobs, nvars,
        // jd, vp, cl, ne, nx, nlam, flmin, ulam, thresh, isd, intr,
        // vnames, maxit)

//        int err = new Fortran().elnet(
//            covUpdating,
//            alpha,
//            yc.elements(),
//            weights,
//            dcdm.elements(),
//            mFlags,
//            penalties,
//            maxFinalFeatures,
//            maxPathFeatures,
//            numLambdas,
//            _mlr,
//            new double[100],
//            convThreshold,
//            standardize ? 1 : 0,
//            maxIterations,
//            outNumFits,
//            outIntercepts,
//            outCoeffs,
//            outCoeffPtrs,
//            outCoeffCnts,
//            outRsq,
//            outLambdas,
//            outNumPasses);
        break;
      case Binomial:
//        binomial = lognet(x, is.sparse, ix, jx, y, weights, offset,
//            alpha, nobs, nvars, jd, vp, cl, ne, nx, nlam, flmin,
//            ulam, thresh, isd, intr, vnames, maxit, kopt, family),
        mods = Lognet.fit(x, y, weights, null, alpha, jd, vp, cl, ne, nx, nlam, flmin, ulam, thresh, isd,intr,null,maxit,kopt, family);
        break;
    }

//    fit = switch(family,, poisson = fishnet(x, is.sparse, ix, jx,
//        y, weights, offset, alpha, nobs, nvars, jd, vp, cl, ne,
//        nx, nlam, flmin, ulam, thresh, isd, intr, vnames, maxit),

//        multinomial = lognet(x, is.sparse, ix, jx, y, weights,
//            offset, alpha, nobs, nvars, jd, vp, cl, ne, nx, nlam,
//            flmin, ulam, thresh, isd, intr, vnames, maxit, kopt,
//            family), cox = coxnet(x, is.sparse, ix, jx, y, weights,
//        offset, alpha, nobs, nvars, jd, vp, cl, ne, nx, nlam,
//        flmin, ulam, thresh, isd, vnames, maxit), mgaussian = mrelnet(x,
//        is.sparse, ix, jx, y, weights, offset, alpha, nobs,
//        nvars, jd, vp, cl, ne, nx, nlam, flmin, ulam, thresh,
//        isd, jsd, intr, vnames, maxit))

    if (lambda == null) {
      mods.fixLambda();
    }

//    fit$nobs = nobs
//    class(fit) = c(class(fit), "glmnet")
//    fit
    return mods;
  }

  double[] toArray(List<Double> list) {
    return list.stream().mapToDouble(Double::doubleValue).toArray();
  }
}

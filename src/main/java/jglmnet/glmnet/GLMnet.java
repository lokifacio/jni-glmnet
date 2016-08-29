package jglmnet.glmnet;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseColumnDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix3D;

import java.util.Arrays;
import java.util.List;

/**
 * Java implementation of R glmnet
 * @author Jorge Peña
 */
public class GLMnet {

  enum Family {Gaussian, Binomial, Poisson, Multinomial, Cox, MGaussian}
  enum LogisticType {Newton, ModifiedNewton}
  enum MultinomialType {Ungrouped, Grouped}

  // Default values taken from glmnet.R
  private double alpha = 1.0;
  private Family family = Family.Binomial;
  private int nlam = 100;
  private Double lambdaMinRatio;
  private DoubleMatrix1D lambda;

  private boolean standardize = true;
  private boolean intercept = true;
  private double thresh = 1e-07;
  private Integer dfmax;
  private Integer pmax;
  // exlude?
  DoubleMatrix1D penaltyFactor; // rep(1, nvars)
  boolean standardizeResponse = false;
  LogisticType logisticType = LogisticType.Newton;
  MultinomialType multinomialType = MultinomialType.Ungrouped;
  int maxit = (int)1e5;

  //        lower.limits = -Inf, upper.limits = Inf, maxit = 1e+05, type.gaussian = ifelse(nvars <
//        500, "covariance", "naive"), type.multinomial = c("ungrouped",
//        "grouped")

  public GLMnet setAlpha(double value) {
    alpha = value;

    if (alpha > 1) {
      System.out.println("Warning: alpha > 1; set to 1");
      alpha = 1;
    } else if (alpha < 0) {
      System.out.println("Warning: alpha > 1; set to 1");
      alpha = 0;
    }

    return this;
  }

  public GLMnet setNLambdas(int value) {
    nlam = value;

    return this;
  }

  public GLMnet setLambdaMinRatio(double value) {
    lambdaMinRatio = value;

    return this;
  }

  public GLMnet setLambdas(DoubleMatrix1D lambdas) {
    lambda = lambdas;

    return this;
  }

  // maximum number of variables allowed to enter largest model
  // (stopping criterion)
  public GLMnet setDFMax(int value) {
    dfmax = value;

    return this;
  }

  //nx = maximum number of variables allowed to enter all models
  //     along path (memory allocation, pmax > dfmax).
  public GLMnet setPMax(int value) {
    pmax = value;

    return this;
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
//    if (any(lower.limits > 0)) {
//      stop("Lower limits should be non-positive")
//    }
//    if (any(upper.limits < 0)) {
//      stop("Upper limits should be non-negative")
//    }
//    lower.limits[lower.limits == -Inf] = -internal.parms$big
//    upper.limits[upper.limits == Inf] = internal.parms$big
//    if (length(lower.limits) < nvars) {
//      if (length(lower.limits) == 1)
//        lower.limits = rep(lower.limits, nvars)
//      else stop("Require length 1 or nvars lower.limits")
//    }
//    else lower.limits = lower.limits[seq(nvars)]
//    if (length(upper.limits) < nvars) {
//      if (length(upper.limits) == 1)
//        upper.limits = rep(upper.limits, nvars)
//      else stop("Require length 1 or nvars upper.limits")
//    }
//    else upper.limits = upper.limits[seq(nvars)]
//    cl = rbind(lower.limits, upper.limits)
//    if (any(cl == 0)) {
//      fdev = glmnet.control()$fdev
//      if (fdev != 0) {
//        glmnet.control(fdev = 0)
//        on.exit(glmnet.control(fdev = fdev))
//      }
//    }
//    storage.mode(cl) = "double"
    DenseColumnDoubleMatrix2D cl = new DenseColumnDoubleMatrix2D(2, nvars);

    for (int c = 0; c < nvars; c++) {
      cl.set(0, c, Double.NEGATIVE_INFINITY);
      cl.set(1, c, Double.POSITIVE_INFINITY);
    }

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
      for(double value : lambda.toArray()) {
        if (value < 0) {
          throw new Exception("Lambdas should be non-negative");
        }
      }
      double[] lambdas = lambda.toArray();
      Arrays.sort(lambdas);

      flmin = 1;
      ulam = lambdas;
      nlam = lambdas.length;
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
}

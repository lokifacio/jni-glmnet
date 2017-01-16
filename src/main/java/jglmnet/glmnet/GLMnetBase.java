package jglmnet.glmnet;

import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author Jorge Peña
 */
public class GLMnetBase implements Serializable {

  // Default values taken from glmnet.R
  protected double alpha;
  protected int nlam;
  protected Double lambdaMinRatio;
  protected List<Double> lambdas;
  protected Family family;
  protected int maxit;
  protected boolean intercept;
  protected boolean standardize;
  protected double thresh;
  protected List<Double> lowerLimits;
  protected List<Double> upperLimits;
  protected DenseDoubleMatrix1D vp;

  public GLMnetBase setAlpha(double value) {
    alpha = value;

    if (alpha > 1) {
      System.err.println("Warning: alpha > 1; set to 1");
      alpha = 1;
    } else if (alpha < 0) {
      System.err.println("Warning: alpha > 1; set to 1");
      alpha = 0;
    }
    return this;
  }

  public GLMnetBase setNLambdas(int value) {
    nlam = value;
    return this;
  }

  public GLMnetBase setLambdaMinRatio(Double value) {
    lambdaMinRatio = value;
    return this;
  }

  public GLMnetBase setLambdas(List<Double> lambdas) {
    this.lambdas = lambdas;
    return this;
  }

  public GLMnetBase setFamily(Family family) {
    this.family = family;
    return this;
  }

  public GLMnetBase setThreshold(double value) {
    thresh = value;
    return this;
  }

  public GLMnetBase setIntercept(boolean value) {
    intercept = value;
    return this;
  }

  public GLMnetBase setStandardize(boolean value) {
    standardize = value;
    return this;
  }

  public GLMnetBase setMaxIter(int n) {
    maxit = n;
    return this;
  }

  public GLMnetBase setLimits(List<Double> lower, List<Double> upper) throws Exception {
    if (Collections.min(lower) > 0) {
      throw new Exception("Lower limits should be non-positive");
    } else {
      lowerLimits = lower;
    }
    if (Collections.max(lower) < 0) {
      throw new Exception("Upper limits should be non-negative");
    } else {
      upperLimits = upper;
    }
    return this;
  }

  public GLMnetBase setPenaltyFactor(DenseDoubleMatrix1D penaltyFactor) {
    vp = penaltyFactor;
    return this;
  }

  public GLMnetBase() {
    alpha = 1.0;
    nlam = 100;
    lambdaMinRatio = null;
    lambdas = null;
    family = Family.Binomial;
    maxit = (int)1e5;
    intercept = true;
    standardize = true;
    thresh = 1e-07;
    lowerLimits = new ArrayList<>();
    upperLimits = new ArrayList<>();
    vp = null;
  }

  public GLMnetBase(GLMnetBase other) {
    alpha          = other.alpha;
    nlam           = other.nlam;
    lambdaMinRatio = other.lambdaMinRatio;
    lambdas = other.lambdas;
    family         = other.family;
    maxit          = other.maxit;
    intercept      = other.intercept;
    standardize    = other.standardize;
    thresh         = other.thresh;
    lowerLimits    = other.lowerLimits;
    upperLimits    = other.upperLimits;
    vp             = other.vp;
  }


}

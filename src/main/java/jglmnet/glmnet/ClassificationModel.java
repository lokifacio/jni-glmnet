package jglmnet.glmnet;


import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;

import java.io.Serializable;

/**
 * Linear model
 *
 * @author Thomas Down
 * @author Jorge Peña
 */

public abstract class ClassificationModel implements Serializable {
  private final double intercept;
  private final DoubleMatrix1D betas;
  private final double lambda;

  ClassificationModel(double intercept, DoubleMatrix1D betas, double lambda) {
    this.betas = betas;
    this.intercept = intercept;
    this.lambda = lambda;
  }

  public double getIntercept() { return intercept; }

  public DoubleMatrix1D getBetas() { return betas; }

  public double getLambda() { return lambda; }

  public double predict(DoubleMatrix1D features) throws Exception {
    if (requiresOffset()) {
      throw new Exception("Required offset value");
    }
    return predict(features, 0);
  }

  public double predict(DoubleMatrix1D features, double offset) {
    return intercept + betas.zDotProduct(features) + offset;
  }

  public DoubleMatrix1D predict(DoubleMatrix2D features) throws Exception {
    return predict(features, null);
  }

  public DoubleMatrix1D predict(DoubleMatrix2D features, DoubleMatrix1D offset) throws Exception {
    DoubleMatrix1D result = DenseDoubleAlgebra.DEFAULT.mult(features, betas);

    if (requiresOffset() && offset == null) {
      throw new Exception("Required offset value");
    }
    if (offset == null) {
      result.assign(x -> intercept + x);
    } else {
      result.assign(offset, (x, y) -> intercept + x + y);
    }

    return result;
  }

  public double response(DoubleMatrix1D features) throws Exception {
    if (requiresOffset()) {
      throw new Exception("Required offset value");
    }
    return response(features, 0);
  }

  public double response(DoubleMatrix1D features, double offset) {
    return invLink(predict(features, offset));
  }

  public DoubleMatrix1D response(DoubleMatrix2D features) throws Exception {
    return response(features, null);
  }

  public DoubleMatrix1D response(DoubleMatrix2D features, DoubleMatrix1D offset) throws Exception {
    DoubleMatrix1D response = predict(features, offset);

    response.assign(this::invLink);

    return response;
  }

  protected abstract boolean requiresOffset();

  public abstract double invLink(double x);
}
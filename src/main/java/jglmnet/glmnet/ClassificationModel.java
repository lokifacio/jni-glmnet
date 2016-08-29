package jglmnet.glmnet;


import cern.colt.matrix.tdouble.DoubleMatrix1D;

/**
 * Linear model
 *
 * @author Thomas Down
 */

public class ClassificationModel {
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

    public double estimate(DoubleMatrix1D features) {
        return logit(intercept + betas.zDotProduct(features));
    }

    public static double logit(double x) {
	return 1.0 / (1.0 + Math.exp(-x));
    }
}
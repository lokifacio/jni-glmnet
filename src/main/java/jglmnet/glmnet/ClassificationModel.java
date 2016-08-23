package jglmnet.glmnet;


import cern.colt.matrix.tdouble.DoubleMatrix1D;

/**
 * Linear model
 *
 * @author Thomas Down
 */

public class ClassificationModel {
    private final double intercept;
    private final DoubleMatrix1D weights;
    private final double lambda;

    ClassificationModel(double intercept, DoubleMatrix1D weights, double lambda) {
      this.weights = weights;
      this.intercept = intercept;
      this.lambda = lambda;
    }

    public double getIntercept() { return intercept; }

    public DoubleMatrix1D getBetas() { return weights; }

    public double getLambda() { return lambda; }

    public double estimate(DoubleMatrix1D features) {
        //return logit(intercept + weights.zDotProduct(features));
        double dot = 0;
        for (int i = 0; i < features.size(); i++) {
            dot += features.get(i)*weights.get(i);
        }

        return logit(intercept + dot);
    }

    private double logit(double x) {
	return 1.0 / (1.0 + Math.exp(-x));
    }
}
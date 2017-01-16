package jglmnet.glmnet;

import cern.colt.matrix.tdouble.*;
import cern.colt.matrix.tdouble.impl.*;

/**
 * Linear model
 *
 * @author Thomas Down
 */

public class RegressionModel {
    private final double intercept;
    private final DoubleMatrix1D weights;

    RegressionModel(double intercept, DoubleMatrix1D weights) {
        this.weights = weights;
        this.intercept = intercept;
    }

    public double getIntercept() { return intercept; }

    public DoubleMatrix1D getBetas() { return weights; }

    public double estimate(DoubleMatrix1D features) {
        //return intercept + weights.zDotProduct(features);
        double dot = 0;
        for (int i = 0; i < features.size(); i++) {
            dot += features.get(i)*weights.get(i);
        }

        return this.intercept + dot;
    }
}
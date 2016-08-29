package jglmnet.glmnet;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DiagonalDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix1D;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

/**
 * Set of models produced by a glmnet learner run.
 *
 * @author Thomas Down
 */

public class ClassificationModelSet {
  private final int numPasses;
  private final int numFits;
  private final double[] intercepts;
  private final double[] coeffs;
  private final int[] coeffPtrs;
  private final int[] coeffCnts;
  private final double[] lambdas;
  private final int columns;
  private final int maxPathFeatures;

  ClassificationModelSet(int numPasses, int numFits,
                         double[] intercepts, double[] coeffs, int[] coeffPtrs, int[] coeffCnts, double[] lambdas,
                         int columns, int maxPathFeatures) {
    this.numPasses = numPasses;
    this.numFits = numFits;
    this.intercepts = intercepts;
    this.coeffs = coeffs;
    this.coeffPtrs = coeffPtrs;
    this.coeffCnts = coeffCnts;
    this.lambdas = lambdas;

    this.columns = columns;
    this.maxPathFeatures = maxPathFeatures;
  }

  public int getNumPasses() {
    return numPasses;
  }

  public int getNumFits() {
    return numFits;
  }

  public List<Double> getLambdas() {
    return  DoubleStream.of(lambdas)
        .mapToObj(Double::valueOf)
        .filter(d -> !Double.isNaN(d))
        .collect(Collectors.toList());
  }

  public ClassificationModel getModel(int i) {
    if (i < 0 || i >= numFits) {
      throw new IllegalArgumentException(String.format("No model %d, allowed range 0-%d", i, numFits - 1));
    }

    DoubleMatrix1D weights = new SparseDoubleMatrix1D(columns);
    for (int j = 0; j < coeffCnts[i]; ++j) {
      weights.set(coeffPtrs[j] - 1, coeffs[i * maxPathFeatures + j]);
    }
    return new ClassificationModel(intercepts[i], weights, lambdas[i]);
  }

  public ClassificationModel getModel(double s) {
    int pos = 0;
    double minErr = Math.abs(s - lambdas[pos]);

    for (int i = 0; i < lambdas.length; ++i) {
      double err =  Math.abs(s - lambdas[i]);
      if (err < minErr) {
        minErr = err;
        pos = i;
      }
    }

    return getModel(pos);
  }

  public void fixLambda() {
    if (lambdas.length > 2) {
      lambdas[0] = Math.exp(2 * Math.log(lambdas[1]) - Math.log(lambdas[2]));
    }
  }

  public double response(DoubleMatrix1D newx, double s) throws Exception {

    if (newx == null) {
      throw new Exception("You need to supply a value for 'newx'");
    }

    return getModel(s).estimate(newx);
  }

  public DoubleMatrix1D response(DoubleMatrix2D newx, double s) throws Exception {

    if (newx == null) {
      throw new Exception("You need to supply a value for 'newx'");
    }

    List<Double> lambdas = getLambdas();
    int left = 0;
    int right = lambdas.size() - 1;

    for (int i = 0; i < lambdas.size(); ++i) {
      double lambda = lambdas.get(i);
      if (lambda >= s) {
        left  = Math.max(left, i);
      }
      if (lambda <= s) {
        right = Math.min(right, i);
      }
    }

    double sfrac = (left == right)?1:(s - lambdas.get(right))/(lambdas.get(left) - lambdas.get(right));
    ClassificationModel leftFit  = getModel(left);
    double leftIntercept = sfrac*leftFit.getIntercept();
    DoubleMatrix1D betas = leftFit.getBetas().copy();

    if (sfrac != 1) {
      ClassificationModel rightFit = getModel(right);
      leftIntercept += (1 - sfrac)*rightFit.getIntercept();

      DoubleMatrix1D rBetas = rightFit.getBetas();
      for (int i = 0; i < newx.columns(); ++i) {
        betas.set(i, sfrac*betas.get(i) + (1-sfrac)*rBetas.get(i));
      }
    }

    final double intercept = leftIntercept;

    DoubleMatrix1D response = DenseDoubleAlgebra.DEFAULT.mult(newx, betas);
    response.assign(v -> ClassificationModel.logit(intercept + v));

    return response;
  }
}
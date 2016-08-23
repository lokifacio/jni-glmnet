package jglmnet.glmnet;

import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

public class TestLognet {
    public static void main(String[] args) {
        double[] y = {
                1,
                0,
                0,
                1
        };

        DenseDoubleMatrix1D target = new DenseDoubleMatrix1D(y);

        int rows = y.length;
        int cols = 2;

        DenseDoubleMatrix2D matrix = new DenseDoubleMatrix2D(rows, cols);
        matrix.set(0, 0, 1);
        matrix.set(1, 0, 0);
        matrix.set(2, 0, 0);
        matrix.set(3, 0, 1);
        matrix.set(0, 1, -21);
        matrix.set(1, 1, 7);
        matrix.set(2, 1, 3);
        matrix.set(3, 1, 9);

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                System.out.printf("%.3g\t", matrix.get(r, c));
            }
            System.out.println();
        }

        double[] w = new double[rows];
        for (int i = 0; i < rows; ++i) {
            w[i] = 1;
        }

        RegressionLearner learner = new RegressionLearner();
        learner.setAlpha(1);
        learner.setStandardize(true);
        learner.setConvThreshold(0.0000001);

        RegressionModelSet mods = learner.learn(target, matrix);
        System.err.printf("numFits=%d, numPasses=%d%n", mods.getNumFits(), mods.getNumPasses());

        for (int i = 0; i < mods.getNumFits(); ++i) {
            RegressionModel mod = mods.getModel(i);
            System.out.println("Model " + i);
            System.out.println("\tIntecept:" + mod.getIntercept());
            System.out.println("\tBetas:\n\t\t" + mod.getBetas().get(0) + "\n\t\t" + mod.getBetas().get(1));
            System.out.println("\tPred 1:" + mod.estimate(new DenseDoubleMatrix1D(new double[]{1, -21})));
            System.out.println("\tPred 0:" + mod.estimate(new DenseDoubleMatrix1D(new double[]{0, 7})));
        }
    }
}
package jglmnet.glmnet;

import cern.colt.matrix.tdouble.impl.DenseColumnDoubleMatrix2D;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class TestLognet {
  public static void main(String[] args) throws IOException {
    InputStream input = TestLognet.class.getClassLoader().getResourceAsStream("binary.csv");
    BufferedReader reader = new BufferedReader(new InputStreamReader(input));

    // Saltamos la cabecera
    String line = reader.readLine();

    int rows = 400;
    int cols = 3;

    DenseColumnDoubleMatrix2D y  = new DenseColumnDoubleMatrix2D(rows, 2);
    DenseColumnDoubleMatrix2D dm = new DenseColumnDoubleMatrix2D(rows, cols);
    int r = 0;
    while ((line = reader.readLine()) != null) {
      String[] row = line.split(",");

      double target = Double.parseDouble(row[0]);
      y.set(r, 0, target);
      y.set(r, 1, 1 - target);

      for (int c = 0; c < cols; c++) {
        dm.set(r, c, Double.parseDouble(row[1 + c]));
      }

      r++;
    }

    for (int i = 0; i < 10; ++i) {
      System.out.printf("%.3g\t", y.get(i, 0));
      for (int c = 0; c < cols; ++c) {
        System.out.printf("%.3g\t", dm.get(i, c));
      }
      System.out.println();
    }

    double[] w = new double[rows];
    for (int i = 0; i < rows; ++i) {
      w[i] = 1;
    }

    double[] penalties = new double[cols];
    for (int i = 0; i < cols; ++i) {
      penalties[i] = 1.0;
    }

    DenseColumnDoubleMatrix2D limits = new DenseColumnDoubleMatrix2D(2, cols);

    for (int c = 0; c < cols; c++) {
      limits.set(0, c, Double.NEGATIVE_INFINITY);
      limits.set(1, c, Double.POSITIVE_INFINITY);
    }

    int maxFinalFeatures = cols + 1;
    int maxPathFeatures = Math.min(maxFinalFeatures * 2, cols);

    int numLambdas = 100;

    double[] outIntercepts = new double[numLambdas];
    double[] outCoeffs = new double[maxPathFeatures * numLambdas];
    int[] outCoeffPtrs = new int[maxPathFeatures];
    int[] outCoeffCnts = new int[numLambdas];
    double[] outDev0 = new double[numLambdas];
    double[] outFdev = new double[numLambdas];
    double[] outLambdas = new double[numLambdas];
    int[] outNumPasses = new int[1];
    int[] outNumFits = new int[1];

    int err = Fortran.lognet(
        1.0,
        1,
        y.elements(),
        new double[rows],
        dm.elements(),
        new int[1],
        penalties,
        limits.elements(),
        maxFinalFeatures,
        maxPathFeatures,
        100,
        0.0001,
        new double[100],
        0.000001,
        1,
        1,
        100000,
        0,
        outNumFits,
        outIntercepts,
        outCoeffs,
        outCoeffPtrs,
        outCoeffCnts,
        outDev0,
        outFdev,
        outLambdas,
        outNumPasses);

    System.out.println("Error: " + err);

    ClassificationModelSet mods = new ClassificationModelSet(outNumPasses[0], outNumFits[0], outIntercepts, outCoeffs, outCoeffPtrs, outCoeffCnts, outLambdas, cols, maxPathFeatures);
    for (int i = 0; i < mods.getNumFits(); ++i) {
      ClassificationModel mod = mods.getModel(i);
      System.out.println("Model " + i);
      System.out.println("\tLambda:" + mod.getLambda());
      System.out.println("\tIntecept:" + mod.getIntercept());
      System.out.println("\tBetas:\n\t\t" + mod.getBetas().toString());
      System.out.println("\tPred 1:" + mod.estimate(dm.viewRow(1)));
      System.out.println("\tPred 0:" + mod.estimate(dm.viewRow(0)));
    }
  }
}
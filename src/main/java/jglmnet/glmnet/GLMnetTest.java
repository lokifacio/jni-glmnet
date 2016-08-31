package jglmnet.glmnet;

import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;

/**
 * @author Jorge Peña
 */
public class GLMnetTest {

  public static void main(String[] args) throws Exception {
    InputStream input = GLMnetTest.class.getClassLoader().getResourceAsStream("binary.csv");
    BufferedReader reader = new BufferedReader(new InputStreamReader(input));

    // Saltamos la cabecera
    String line = reader.readLine();

    int rows = 400;
    int cols = 3;

    DenseDoubleMatrix1D y  = new DenseDoubleMatrix1D(rows);
    DenseDoubleMatrix2D dm = new DenseDoubleMatrix2D(rows, cols);
    int r = 0;
    while ((line = reader.readLine()) != null) {
      String[] row = line.split(",");

      double target = Double.parseDouble(row[0]);
      y.set(r, target);

      for (int c = 0; c < cols; c++) {
        dm.set(r, c, Double.parseDouble(row[1 + c]));
      }

      r++;
    }

    for (int i = 0; i < 10; ++i) {
      System.out.printf("%.3g\t", y.get(i));
      for (int c = 0; c < cols; ++c) {
        System.out.printf("%.3g\t", dm.get(i, c));
      }
      System.out.println();
    }

    DenseDoubleMatrix1D weights = new DenseDoubleMatrix1D(rows);
    double w = 0.25;
    for (int i = 0; i < rows; ++i) {
      weights.set(i, w);
      w += 0.25;
      if (w > 1) {
        w = 0.25;
      }
    }


    GLMnet GLMnet = new GLMnet()
        .setLambdas(Arrays.asList(0.0, 0.05));

    ClassificationModelSet mods = GLMnet.fit(dm, y, weights);

    for (int i = 0; i < mods.getNumFits(); ++i) {
      ClassificationModel mod = mods.getModel(i);
      System.out.println("Model " + i);
      System.out.println("\tLambda: " + mod.getLambda());
      System.out.println("\tIntecept: " + mod.getIntercept());
      System.out.println("\tBetas:\n\t\t" + mod.getBetas().toString());
      System.out.println("\tPred 0: " + mod.estimate(dm.viewRow(0)));
      System.out.println("\tPred 1: " + mod.estimate(dm.viewRow(1)));
    }
  }
}
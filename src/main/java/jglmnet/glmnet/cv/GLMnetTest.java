package jglmnet.glmnet.cv;

import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import jglmnet.glmnet.ClassificationModel;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

    DenseDoubleMatrix1D y = new DenseDoubleMatrix1D(rows);
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


    int nfolds = 10;
    List<Integer> foldid = new ArrayList<>(rows);

    int fold = 0;
    for (int i = 0; i < rows; ++i) {
      foldid.add(fold);
      fold = (fold + 1) % nfolds;
    }

    //long sstart = System.currentTimeMillis();

    double numTests = 100;
    long start = System.currentTimeMillis();
    for (int i = 0; i < numTests; ++i) {
      GLMnet glmnet = new GLMnet()
          .setFoldId(foldid)
          .setParallel(false);

      Model model = glmnet.fit(dm, y, weights);

      if (i == numTests - 1) {
        System.out.println("\tLambda.min: " + model.lambdaMin);
        System.out.println("\tLambda.1se: " + model.lambda1se);
        System.out.println();

        for (double s : Arrays.asList(model.lambdaMin, model.lambda1se)) {
          ClassificationModel mod = model.coef(s);
          System.out.println("\tLambda: " + mod.getLambda());
          System.out.println("\tIntecept: " + mod.getIntercept());
          System.out.println("\tBetas:\n\t\t" + mod.getBetas().toString());
          System.out.println("\tPred 0: " + mod.estimate(dm.viewRow(0)));
          System.out.println("\tPred 1: " + mod.estimate(dm.viewRow(1)));
          System.out.println();
        }
      }
    }
    long end = System.currentTimeMillis();

    System.out.println("Avg time: " + (end - start)/numTests + " ms");
  }
}
package jglmnet.glmnet.cv;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseColumnDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
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
public class BinomialTest {

  public static void main(String[] args) throws Exception {
    InputStream input = BinomialTest.class.getClassLoader().getResourceAsStream("binary.csv");
    BufferedReader reader = new BufferedReader(new InputStreamReader(input));

    // Saltamos la cabecera
    String line = reader.readLine();

    int rows = 400;
    int nvars = 3;

    DoubleMatrix1D y = new DenseDoubleMatrix1D(rows);
    DoubleMatrix2D x = new DenseColumnDoubleMatrix2D(rows, nvars);
    DenseDoubleMatrix1D weights = new DenseDoubleMatrix1D(rows);
    DenseDoubleMatrix1D offset = new DenseDoubleMatrix1D(rows);

    int r = 0;
    while ((line = reader.readLine()) != null) {
      String[] row = line.split(",");

      double target = Double.parseDouble(row[0]);
      y.set(r, target);

      for (int c = 0; c < nvars; c++) {
        x.set(r, c, Double.parseDouble(row[1 + c]));
      }

      weights.set(r, Double.parseDouble(row[nvars + 1]));
      offset.set(r, Double.parseDouble(row[nvars + 2]));

      r++;
    }

    for (int i = 0; i < 10; ++i) {
      System.out.printf("%.3g\t", y.get(i));
      for (int c = 0; c < nvars; ++c) {
        System.out.printf("%.3g\t", x.get(i, c));
      }
      System.out.println();
    }

    int nfolds = 10;
    List<Integer> foldid = new ArrayList<>(rows);

    int fold = 0;
    for (int i = 0; i < rows; ++i) {
      foldid.add(fold + 1);
      fold = (fold + 1) % nfolds;
    }

    //long sstart = System.currentTimeMillis();

    double numTests = 200;
    long start = System.currentTimeMillis();
    for (int i = 0; i < numTests; ++i) {
      GLMnet glmnet = new GLMnet()
          .setFoldId(foldid)
          .setParallel(false);

      Model model = glmnet.fit(x, y, weights, offset);

      if (i == numTests - 1) {
        System.out.println("\tLambda.min: " + model.lambdaMin);
        System.out.println("\tLambda.1se: " + model.lambda1se);
        System.out.println();

        for (double s : Arrays.asList(model.lambdaMin, model.lambda1se)) {
          ClassificationModel mod = model.coef(s);
          System.out.println("\tLambda: " + mod.getLambda());
          System.out.println("\tIntecept: " + mod.getIntercept());
          System.out.println("\tBetas:\n\t\t" + mod.getBetas().toString());
          System.out.println("\tPred 0: " + mod.response(x.viewRow(0)));
          System.out.println("\tPred 1: " + mod.response(x.viewRow(1)));
          System.out.println();
        }
      }
    }
    long end = System.currentTimeMillis();

    System.out.println("Avg time: " + (end - start)/numTests + " ms");
  }
}
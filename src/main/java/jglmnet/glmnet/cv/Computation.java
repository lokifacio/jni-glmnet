package jglmnet.glmnet.cv;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import org.apache.commons.math3.util.Pair;

import java.util.List;

/**
 * Linea Directa Aseguradora
 * Proyecto: GEO
 * Modulo: glmnet
 * Creado: 16/01/2017
 *
 * @author ldajpp1
 */
class Computation {

  public DoubleMatrix2D cvram;
  public DoubleMatrix1D weights;
  public int N;

  Computation(DoubleMatrix2D cvram, DoubleMatrix1D weights, int N) {
    this.cvram   = cvram;
    this.weights = weights;
    this.N = N;
  }

  public static Computation cvcompute(DoubleMatrix2D mat, DoubleMatrix1D weights, List<Integer> foldid, int nlam) {
    int nfolds = Folds.numFolds(foldid);
    double wisum[] = new double[nfolds];
    for (int i = 0; i < foldid.size(); ++i) {
      int fold = foldid.get(i) - 1;
      wisum[fold] += weights.get(i);
    }

    DoubleMatrix2D outmat = new DenseDoubleMatrix2D(nfolds, mat.columns());
    outmat.assign(0);

    for (int l = 0; l < mat.columns(); ++l) {
      for (int i = 0; i < foldid.size(); ++i) {
        int fold = foldid.get(i) - 1;
        double cell = mat.get(i, l);
        if (Double.isInfinite(cell)) {
          cell = Double.NaN;
        }
        outmat.set(fold, l, outmat.get(fold, l) + cell*weights.get(i));
      }
    }

    for (int i = 0; i < nfolds; ++i) {
      final double foldWeight = wisum[i];
      outmat.viewRow(i).assign(v -> v/foldWeight);
    }

    return new Computation(outmat, new DenseDoubleMatrix1D(wisum), outmat.rows());
  }
}

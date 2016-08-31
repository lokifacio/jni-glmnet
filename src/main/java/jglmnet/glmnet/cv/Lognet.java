package jglmnet.glmnet.cv;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import jglmnet.glmnet.ClassificationModelSet;
import jglmnet.glmnet.Classifiers;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.util.Pair;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

/**
 * @author Jorge Peña
 */
public class Lognet {

  //function (outlist, lambda, x, y, weights, offset, foldid, type.measure,
 //           grouped, keep = FALSE)
  public static Measures evaluate(List<ClassificationModelSet> outlist,
                                  List<Double> lambda,
                                  DoubleMatrix2D x,
                                  DoubleMatrix1D y,
                                  DoubleMatrix1D weights,
                                  List<Integer> foldid,
                                  GLMnet.MeasureType type) {


    GLMnet.MeasureType measureType = type;


    if (type == GLMnet.MeasureType.Default) {
      type = GLMnet.MeasureType.Deviance;
    }
    boolean grouped = true; // TODO: pass as parameter

    //System.err.println("Only 'deviance', 'class', 'auc', 'mse' or 'mae'  available for binomial models; 'deviance' used");
    if (measureType == GLMnet.MeasureType.Default) {
      measureType = GLMnet.MeasureType.Deviance;
    }

    final double prob_min = 1e-05;
    final double prob_max = 1 - prob_min;

    //TODO: Validate for nc > 2
    int nc = Classifiers.getNumberOfClasses(y);

    long N = (int) y.size();

    int nfolds = Folds.numFolds(foldid);

    if ((N / nfolds < 10) && measureType == GLMnet.MeasureType.AUC) {
      System.err.println("Warning: Too few (< 10) observations per fold for type.measure='auc' in cv.lognet; changed to type.measure='deviance'. Alternatively, use smaller value for nfolds");
      measureType = GLMnet.MeasureType.Deviance;
    }

    if ((N / nfolds < 3) && grouped) {
      System.err.println("Warning: Option grouped=FALSE enforced in cv.glmnet, since < 3 observations per fold");
      grouped = false;
    }

    DoubleMatrix1D offset = null;

    boolean isOffset = offset != null;
    
    final double mlami = outlist.stream()
        .mapToDouble(fit ->
            fit.getLambdas().stream()
                .mapToDouble(Double::valueOf)
                .min().orElse(0))
        .max().orElse(0);

    List<Double> which_lam = lambda.stream().filter(l -> l >= mlami).collect(Collectors.toList());

    //predmat = matrix(NA, nrow(y), length(lambda))
    DenseDoubleMatrix2D predmat = new DenseDoubleMatrix2D((int)y.size(), lambda.size());
    predmat.assign(Double.NaN);

    int nlam = which_lam.size();

    for (int fold = 0; fold < nfolds; ++fold) {
      ClassificationModelSet fitobj = outlist.get(fold);
      Sample test = Folds.testSamples(foldid, fold, x, y, weights, offset);

      for (int l = 0; l < which_lam.size(); ++l) {
        double s = lambda.get(l);
        try {
          DoubleMatrix1D response = fitobj.response(test.x, s);
          for (int r = 0; r < response.size(); ++r) {
            predmat.set(test.pos.get(r), l, response.get(r));
          }
        } catch (Exception e) {
          e.printStackTrace();
        }
      }
    }

    long[] lambdaN = new long[lambda.size()];
    DoubleMatrix2D cvraw = null;// = new DenseDoubleMatrix2D(nfolds, lambda.size());
    if (type == GLMnet.MeasureType.AUC) {
//      DenseDoubleMatrix2D good = new DenseDoubleMatrix2D(nfolds, lambda.size());
//
//      for (int fold = 0; fold < nfolds; ++fold) {
//        for (int s = 0; s < nlams[fold]; ++s) {
//          good.set(fold, s, 1);
//        }
//
//        which = foldid == i
//        for (int j = 0; j < nlams[fold]; --j) {
//          cvraw[i, j] = auc.mat(y[which, ], predmat[which,
//              j], weights[which])
//        }
//      }
//      N = apply(good, 2, sum)
//      weights = tapply(weights, foldid, sum)
    }
    else {
      //ywt = apply(y, 1, sum) // Todavia no tengo muy claro que hace esto
      //y = y/ywt
      //weights = weights * ywt

      for (int c = 0; c < lambda.size(); ++c) {
        lambdaN[c] = N - Arrays.stream(predmat.viewColumn(c).toArray()).filter(Double::isNaN).count();
      }

      switch (measureType){
        case MSE:
          //(y[, 1] - (1 - predmat))^2 + (y[, 2] - predmat)^2,
          break;
        case MAE:
          //abs(y[, 1] - (1 - predmat)) + abs(y[, 2] - predmat)
          break;
        case Deviance:
//          predmat = pmin(pmax(predmat, prob_min), prob_max)
          predmat.assign(v -> Math.min(Math.max(v, prob_min), prob_max));
          final DoubleMatrix1D ly = y.copy().assign(v -> (v == 0)?0:Math.log(v));
          DoubleMatrix2D lp = predmat.copy();
          for (int r = 0; r < lp.rows(); ++r) {
            final double ly_r = ly.get(r);
            if (y.get(r) > 0) {
              lp.viewRow(r).assign(v -> 2 * (ly_r - Math.log(v)));
            } else {
              lp.viewRow(r).assign(v -> 2 * (ly_r - Math.log(1 - v)));
            }
          }
          cvraw = lp;
          break;
        case Class:
          //y[, 1] * (predmat > 0.5) + y[, 2] * (predmat <= 0.5)y[, 1] * (predmat > 0.5) + y[, 2] * (predmat <= 0.5)
          break;
      }

      //TODO: Validate grouped = False

      if (grouped) {
        Pair<DoubleMatrix2D, double[]> cvob = cvcompute(cvraw, weights, foldid, nlam);
        cvraw = cvob.getFirst();
        weights = new DenseDoubleMatrix1D(cvob.getSecond());
        lambdaN = new long[nlam];
        for (int i = 0; i < nlam; ++i) {
          lambdaN[i] = nfolds;
        }
      }
    }


    Measures measures = new Measures();

    Mean m = new Mean();
    for (int i = 0; i < nlam; ++i) {
      double w[] = weights.toArray();
      double c[] = cvraw.viewColumn(i).toArray();

      final double center = m.evaluate(c, w);

      double[] values = DoubleStream.of(c).map(d -> Math.pow(d - center, 2)).toArray();
      measures.cvm.add(center);
      measures.cvsd.add(Math.sqrt(m.evaluate(values, w)/(lambdaN[i] - 1)));
    }
     measures.type = type;

    //if (keep)
//      out$fit.preval = predmat

    return measures;
  }

  static Pair<DoubleMatrix2D, double[]> cvcompute(DoubleMatrix2D mat, DoubleMatrix1D weights, List<Integer> foldid, int nlam) {
    int nfolds = Folds.numFolds(foldid);
    double wisum[] = new double[nfolds];
    for (int i = 0; i < foldid.size(); ++i) {
      int fold = foldid.get(i);
      wisum[fold] += weights.get(i);
    }

    DoubleMatrix2D outmat = new DenseDoubleMatrix2D(nfolds, mat.columns());
    outmat.assign(0);

    DoubleMatrix2D good = new DenseDoubleMatrix2D(nfolds, mat.columns());
    outmat.assign(0);

    mat.assign(v -> Double.isInfinite(v)?Double.NaN:v);

    for (int l = 0; l < mat.columns(); ++l) {
      for (int i = 0; i < foldid.size(); ++i) {
        int fold = foldid.get(i);
        outmat.set(fold, l, outmat.get(fold, l) + mat.get(i, l)*weights.get(i));
      }
    }

    for (int i = 0; i < nfolds; ++i) {
      final double foldWeight = wisum[i];
      outmat.viewRow(i).assign(v -> v/foldWeight);
    }

    return new Pair<>(outmat, wisum);
  }
}

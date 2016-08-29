package jglmnet.glmnet.cv;

import cern.colt.function.tdouble.DoubleFunction;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import jglmnet.glmnet.ClassificationModel;
import jglmnet.glmnet.ClassificationModelSet;
import jglmnet.glmnet.Classifiers;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author Jorge Peña
 */
public class Lognet {

  public static class Measures {
    List<Double> cvm;
    List<Double> cvsd;
    GLMnet.MeasureType type;
  }

  //function (outlist, lambda, x, y, weights, offset, foldid, type.measure,
 //           grouped, keep = FALSE)
  public static Measures evaluate(ClassificationModelSet[] outlist,
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

    switch (measureType) {
      case MSE:
      case MAE:
      case Deviance:
      case AUC:
      case Class:
        break;
      default:
        System.err.println("Only 'deviance', 'class', 'auc', 'mse' or 'mae'  available for binomial models; 'deviance' used");
        measureType = GLMnet.MeasureType.Deviance;
    }

    final double prob_min = 1e-05;
    final double prob_max = 1 - prob_min;

//    nc = dim(y)
//    if (is.null(nc)) {
//    y = as.factor(y)
//    ntab = table(y)
//    nc = as.integer(length(ntab))
//    y = diag(nc)[as.numeric(y), ]
//  }
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

    //TODO:
//    mlami=max(sapply(outlist,function(obj)min(obj$lambda)))
    final double mlami = Stream.of(outlist)
        .mapToDouble(fit ->
            fit.getLambdas().stream()
                .mapToDouble(Double::valueOf)
                .min().orElse(0))
        .max().orElse(0);

    List<Double> which_lam = lambda.stream().filter(l -> l >= mlami).collect(Collectors.toList());

    //predmat = matrix(NA, nrow(y), length(lambda))
    DenseDoubleMatrix2D predmat = new DenseDoubleMatrix2D((int)y.size(), lambda.size());
    predmat.assign(Double.NaN);

    int[] nlams = new int[nfolds];

    int total = 0;
    for (int fold = 0; fold < nfolds; ++fold) {
      ClassificationModelSet fitobj = outlist[fold];
      Sample test = Folds.testSamples(foldid, fold, x, y, weights, offset);

      for (int l = 0; l < which_lam.size(); ++l) {
        double s = lambda.get(l);
        try {
          DoubleMatrix1D response = fitobj.response(test.x, s);
          for (int r = 0; r < response.size(); ++r) {
            predmat.set(total + r, l, response.get(r));
          }
        } catch (Exception e) {
          e.printStackTrace();
        }
      }

      total += test.x.rows();

      int nlami = which_lam.size();
      nlams[fold] = nlami;
    }

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

      long[] lambdaN = new long[lambda.size()];

      for (int c = 0; c < lambda.size(); ++c) {
        lambdaN[c] = N - Arrays.stream(predmat.viewColumn(c).toArray()).filter(Double::isNaN).count();
      }

      switch (type){
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
              lp.viewRow(r).assign(v -> 2 * (ly_r - Math.log(1 - v)));
            } else {
              lp.viewRow(r).assign(v -> 2 * (ly_r - Math.log(v)));
            }
          }

          cvraw = lp;
//          lp = y[, 1] * log(1 - predmat) + y[, 2] * log(predmat)
          //DoubleMatrix1D ly =
//          ly = log(y)
//          ly[y == 0] = 0
//          ly = drop((y * ly) %*% c(1, 1))
//          2 * (ly - lp)
          break;
        case Class:
          //y[, 1] * (predmat > 0.5) + y[, 2] * (predmat <= 0.5)y[, 1] * (predmat > 0.5) + y[, 2] * (predmat <= 0.5)
          break;
      }


      if (grouped) {
        DoubleMatrix2D cvob = cvcompute(cvraw, weights, foldid, nlams);
//        cvraw = cvob$cvraw
//        weights = cvob$weights
//        N = cvob$N
      }
    }

    Measures measures = new Measures();
//    measures.cvm = apply(cvraw, 2, weighted.mean, w = weights, na.rm = TRUE)
//    measures.cvsd = sqrt(apply(scale(cvraw, cvm, FALSE)^2, 2, weighted.mean,
//        w = weights, na.rm = TRUE)/(N - 1))
//    measures.type = type;

    //if (keep)
//      out$fit.preval = predmat
    return measures;
  }

  static DoubleMatrix2D cvcompute(DoubleMatrix2D mat, DoubleMatrix1D weights, List<Integer> foldid, int[] nlams) {
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
        outmat.set(fold, l, outmat.get(fold, l) + mat.get(i, l));
      }
    }

    for (int i = 0; i < nfolds; ++i) {
      final double foldWeight = wisum[i];
      outmat.viewRow(i).assign(v -> v/foldWeight);
    }
    //for (int i = 0; )
//      double weightedMean = 0;
//      for (int c = 0; c < mat.columns(); ++c) {
//        weightedMean += mat.get(fo)
//      }
      //mati = mat[foldid == i, , drop = FALSE]
      //wi = weights[foldid == i]
      //outmat[i, ] = apply(mati, 2, weighted.mean, w = wi, na.rm = TRUE)
      //good[i, seq(nlams[i])] = 1
//    }
//    N = apply(good, 2, sum)
//    list(cvraw = outmat, weights = wisum, N = N)
    return mat;
  }
}

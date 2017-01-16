package jglmnet.glmnet.cv;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseColumnDoubleMatrix2D;
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
public class Fishnet {

  //function (outlist, lambdas, x, y, weights, offset, foldid, type.measure,
  //          grouped, keep = FALSE)
  public static Measures evaluate(List<ClassificationModelSet> outlist,
                                  List<Double> lambda,
                                  DoubleMatrix2D x,
                                  DoubleMatrix1D y,
                                  DoubleMatrix1D weights,
                                  DoubleMatrix1D offset,
                                  List<Integer> foldid,
                                  MeasureType type,
                                  boolean keep) {


    MeasureType measureType = MeasureType.Deviance;

    switch (type) {
      case MSE:
      case MAE:
      case Deviance:
        measureType = type;
        break;
      case Default:
        break;
      default:
        System.err.println("Warning: Only 'deviance', 'mse' or 'mae'  available for Poisson models; 'deviance' used");
    }

    final double mlami = outlist.stream()
        .mapToDouble(fit ->
            fit.getLambdas().stream()
                .mapToDouble(Double::valueOf)
                .min().orElse(0))
        .max().orElse(0);

    List<Double> which_lam = lambda.stream().filter(l -> l >= mlami).collect(Collectors.toList());

    DenseDoubleMatrix2D predmat = new DenseDoubleMatrix2D((int)y.size(), lambda.size());
    predmat.assign(Double.NaN);

    int nlam = which_lam.size();

    int nfolds = Folds.numFolds(foldid);

    for (int fold = 0; fold < nfolds; ++fold) {
      ClassificationModelSet fitobj = outlist.get(fold);
      Sample test = Folds.testSamples(foldid, fold, x, y, weights, offset);

      for (int l = 0; l < which_lam.size(); ++l) {
        double s = lambda.get(l);
        try {
          DoubleMatrix1D prediction = fitobj.predict(test.x, test.o, s);
          for (int r = 0; r < prediction.size(); ++r) {
            predmat.set(test.pos.get(r), l, prediction.get(r));
          }
        } catch (Exception e) {
          e.printStackTrace();
        }
      }
    }

    boolean grouped = true; // TODO: pass as parameter

    DoubleMatrix2D cvraw = new DenseColumnDoubleMatrix2D((int)y.size(), nlam);
    for (int i = 0; i < nlam; ++i) {
      cvraw.viewColumn(i).assign(predmat.viewColumn(i));
    }

    switch (measureType) {
      case MSE:
        for (int i = 0; i < nlam; ++i) {
          cvraw.viewColumn(i).assign(y, (vx, vy) -> Math.pow(vy - Math.exp(vx), 2));
        }
        break;
      case MAE:
        for (int i = 0; i < nlam; ++i) {
          cvraw.viewColumn(i).assign(y, (vx, vy) -> Math.abs(vy - Math.exp(vx)));
        }
        break;
      case Deviance:
        for (int i = 0; i < nlam; ++i) {
          cvraw.viewColumn(i).assign(y, (vx, vy) -> {
            double deveta = vy * vx - Math.exp(vx);
            double devy   = vy == 0?0:vy * Math.log(vy) - vy;

            return 2 * (devy - deveta);
          });
        }
        break;
    }

    if (y.size()/nfolds < 3 && grouped) {
      System.out.println("Option grouped=false enforced in cv.glmnet, since < 3 observations per fold");
      grouped = false;
    }

    int N = (int)y.size();
    if (grouped) {
      Computation cvob = Computation.cvcompute(cvraw, weights, foldid, nlam);
      cvraw = cvob.cvram;
      weights = cvob.weights;
      N = cvob.N;
    }

    long[] lambdaN = new long[lambda.size()];

    for (int i = 0; i < nlam; ++i) {
      lambdaN[i] = N;
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
}

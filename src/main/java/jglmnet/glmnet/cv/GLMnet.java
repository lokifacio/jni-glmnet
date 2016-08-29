package jglmnet.glmnet.cv;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import javafx.util.Pair;
import jglmnet.glmnet.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * @author Jorge Peña
 */
public class GLMnet {

//  function (x, y, weights, offset = NULL, lambda = NULL, type.measure = c("mse",
//                "deviance", "class", "auc", "mae"), nfolds = 10, foldid,
//  grouped = TRUE, keep = FALSE, parallel = FALSE, ...)

  public enum MeasureType {Default, MSE, Deviance, Class, AUC, MAE}

  private List<Double> lambda;
  MeasureType measureType = MeasureType.Default;
  private boolean parallel = false;

  List<Integer> foldid;
  int nfolds = 10; //TODO
  boolean keep = false; //TODO
  boolean grouped = true;


  public GLMnet setMeasureType(MeasureType type) {
    measureType = type;

    return this;
  }

  public Pair<Double, Double> getMin(List<Double> lambda, List<Double> cvm, List<Double> cvsd) {
    double cvmin = Collections.min(cvm);
    double lambdaMin = 0;
    double lambda1se = 0;
    double semin = -1;

    double last = lambda.get(0);
    for (int i = 0; i < lambda.size(); ++i) {
      if (cvm.get(i) <= cvmin) {
        lambdaMin = lambda.get(i);
        semin = cvm.get(i) + cvsd.get(i);
      }
      if (cvm.get(i) <= semin) {
        lambda1se = lambda.get(i);
      }
      assert (last <= lambda.get(i));
      last = lambda.get(i);
    }

    return new Pair<>(lambdaMin, lambda1se);
  }

  public class cvFit {
    Lognet.Measures measures;
    double lambdaMin;
    double lambda1se;
  }

  // Posibles parametros adicionales: "type.measure", "nfolds", "foldid", "grouped", "keep"
  public cvFit fit
  (DoubleMatrix2D x,
   DoubleMatrix1D y,
   DoubleMatrix1D weights //TODO: Pesos por defecto: 1
  ) throws Exception {


    if (lambda != null && lambda.size() < 2) {
      throw new Exception("Need more than one value of lambda for cv.GLMnet");
    }

    int N = x.rows();

    jglmnet.glmnet.GLMnet glmnet = new jglmnet.glmnet.GLMnet();//TODO: Configurar
    ClassificationModelSet glmnetObject = glmnet.fit(x, y, weights);
    List<Double> lambdas = glmnetObject.getLambdas();

    boolean isOffset = false; //TODO: configurar segun los parametros

    // ###Next line is commented out so each call generates its own lambda sequence
    // # lambda=glmnet.object$lambda

    //TODO: habria q pasarle los parametros
//    if (inherits(glmnet.object, "multnet") && !glmnet.object$grouped) {
//      nz = predict(glmnet.object, type = "nonzero")
//      nz = sapply(nz, function(x) sapply(x, length))
//      nz = ceiling(apply(nz, 1, median))
//    }
//    else {
//      nz = sapply(predict(glmnet.object, type = "nonzero"),length)
//    }

    if (foldid == null) {
      foldid = new ArrayList<>(N);
      int fold = 0;
      for (int i = 0; i < N; ++i) {
        foldid.add(fold);
        fold = (fold + 1) % nfolds;
      }
      //Collections.shuffle(foldid);
    } else {
      nfolds = Collections.max(foldid);
    }

    if (nfolds < 3) {
      throw new Exception("nfolds must be bigger than 3; nfolds=10 recommended");
    }

    ClassificationModelSet[] outlist = new ClassificationModelSet[nfolds];

    if (parallel) {
//        outlist = foreach(i = seq(nfolds), .packages = c("glmnet")) %dopar%
//            {
//                which = foldid == i
//        if (is.matrix(y))
//          y_sub = y[!which, ]
//      else y_sub = y[!which]
//        if (is.offset)
//          offset_sub = as.matrix(offset)[!which, ]
//      else offset_sub = NULL
//        glmnet(x[!which, , drop = FALSE], y_sub, lambda = lambda,
//            offset = offset_sub, weights = weights[!which],
//             ...)
//    }
    } else {
      for (int fold = 0; fold < nfolds; ++fold) {
        jglmnet.glmnet.GLMnet glmnet_i = new jglmnet.glmnet.GLMnet();

        Sample sample = Folds.trainSamples(foldid, fold, x, y, weights, null);

        if (isOffset) {
          //TODO: Subsample offset
          //glmnet.setOffset
        }

        outlist[fold] = glmnet_i.fit(sample.x, sample.y, sample.w);
      }
    }

    //TODO: tener en cuenta el resto de familias

    cvFit fit = new cvFit();

    //fit.measures == csvstuff
    fit.measures = Lognet.evaluate(outlist, lambdas, x, y, weights, foldid, measureType);

//      lambda = glmnet.object$lambda


    List<Double> cvm  = fit.measures.cvm;
    List<Double> cvsd = fit.measures.cvsd;
    //TODO
//    cvm = cvstuff$cvm
//    cvsd = cvstuff$cvsd
//    nas = is.na(cvsd)
//    if (any(nas)) {
//      lambda = lambda[!nas]
//      cvm = cvm[!nas]
//      cvsd = cvsd[!nas]
//      nz = nz[!nas]
//    }

//    out = list(lambda = lambda, cvm = cvm, cvsd = cvsd, cvup = cvm +
//        cvsd, cvlo = cvm - cvsd, nzero = nz, name = cvname, glmnet.fit = glmnet.object)
//
//    if (keep) {
//      out = c(out, list(fit.preval = cvstuff$fit.preval, foldid = foldid))
//    }

    Pair<Double, Double> lamin;

    if (fit.measures.type ==  MeasureType.AUC) {
      lamin = getMin(lambda, cvm.stream().map(d -> -d).collect(Collectors.toList()), cvsd);
    } else {
      lamin = getMin(lambda, cvm, cvsd);
    }

    fit.lambdaMin = lamin.getKey();
    fit.lambda1se = lamin.getValue();

    return fit;
  }
}

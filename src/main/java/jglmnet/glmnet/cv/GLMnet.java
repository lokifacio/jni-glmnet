package jglmnet.glmnet.cv;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import jglmnet.glmnet.ClassificationModelSet;
import jglmnet.glmnet.Family;
import jglmnet.glmnet.GLMnetBase;
import org.apache.commons.math3.util.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author Jorge Peña
 */
public class GLMnet extends GLMnetBase{
//  function (x, y, weights, offset = NULL, lambda = NULL, type.measure = c("mse",
//                "deviance", "class", "auc", "mae"), nfolds = 10, foldid,
//  grouped = TRUE, keep = FALSE, parallel = FALSE, ...)

  public enum MeasureType {Default, MSE, Deviance, Class, AUC, MAE}

  private List<Double> lambda;
  private int nfolds = 10;
  private List<Integer> foldid;
  private MeasureType measureType = MeasureType.Default;
  private boolean parallel = false;
  private boolean keep = false; //TODO
  private boolean grouped = true; //TODO

  @Override
  public GLMnet setAlpha(double value) {
    super.setAlpha(value);
    return this;
  }

  @Override
  public GLMnet setNLambdas(int value) {
    super.setNLambdas(value);
    return this;
  }

  @Override
  public GLMnet setLambdaMinRatio(Double value) {
    super.setLambdaMinRatio(value);
    return this;
  }

  @Override
  public GLMnet setLambdas(List<Double> lambdas) {
    super.setLambdas(lambdas);
    return this;
  }

  @Override
  public GLMnet setFamily(Family family) {
    super.setFamily(family);
    return this;
  }

  @Override
  public GLMnet setThreshold(double value) {
    super.setThreshold(value);
    return this;
  }

  @Override
  public GLMnet setIntercept(boolean value) {
    super.setIntercept(value);
    return this;
  }

  @Override
  public GLMnet setStandardize(boolean value) {
    super.setStandardize(value);
    return this;
  }

  @Override
  public GLMnet setMaxIter(int n) {
    super.setMaxIter(n);
    return this;
  }

  @Override
  public GLMnet setLimits(List<Double> lower, List<Double> upper) throws Exception {
    super.setLimits(lower, upper);
    return this;
  }

  public GLMnet setNFolds(int nfolds) {
    this.nfolds = nfolds;
    return this;
  }

  public GLMnet setFoldId(List<Integer> foldId) {
    this.foldid = foldId;
    return this;
  }

  public GLMnet setMeasureType(MeasureType type) {
    measureType = type;
    return this;
  }

  public GLMnet setParallel(boolean value) {
    parallel = value;
    return this;
  }

  public GLMnet() {
  }

  public GLMnet(GLMnetBase other) {
    super(other);
  }

  // Posibles parametros adicionales: "type.measure",   "grouped", "keep"
  public Model fit
  (DoubleMatrix2D x,
   DoubleMatrix1D y,
   DoubleMatrix1D weights //TODO: Pesos por defecto: 1
  ) throws Exception {
    if (lambda != null && lambda.size() < 2) {
      throw new Exception("Need more than one value of lambda for cv.GLMnet");
    }

    int N = x.rows();

    jglmnet.glmnet.GLMnet glmnet = new jglmnet.glmnet.GLMnet(this); // Copies common params

    ClassificationModelSet glmnetObject = glmnet.fit(x, y, weights);
    List<Double> lambdas = glmnetObject.getLambdas();

    boolean isOffset = false; //TODO: configurar segun los parametros

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
      foldid = Folds.generateFoldIds(nfolds, N);
    } else {
      nfolds = Folds.numFolds(foldid);
    }

    if (nfolds < 3) {
      throw new Exception("nfolds must be bigger than 3; nfolds=10 recommended");
    }

    List<ClassificationModelSet> outlist;

    if (parallel) {
      //List<Pair<Integer, ClassificationModelSet>> vas = IntStream.range(0, nfolds)
      outlist = IntStream.range(0, nfolds)
          .boxed()
          .parallel()
          .map(fold -> {
            jglmnet.glmnet.GLMnet glmnet_i = new jglmnet.glmnet.GLMnet();

            Sample sample = Folds.trainSamples(foldid, fold, x, y, weights, null);

            if (isOffset) {
              //TODO: Subsample offset
              //glmnet.setOffset
            }

            ClassificationModelSet modelSet = null;
            try {
              modelSet = glmnet_i.fit(sample.x, sample.y, sample.w);
            } catch (Exception e) {
              e.printStackTrace();
            }
            return new Pair<>(fold, modelSet);
          })
          .collect(Collectors.toList()) // serializes model results
          .stream()
          .sorted((p1, p2) -> Integer.compare(p1.getFirst(), p2.getFirst()))
          .map(p -> p.getSecond())
          .collect(Collectors.toList());
    } else {
      outlist = new ArrayList<>(nfolds);

      for (int fold = 0; fold < nfolds; ++fold) {
        jglmnet.glmnet.GLMnet glmnet_i = new jglmnet.glmnet.GLMnet();

        Sample sample = Folds.trainSamples(foldid, fold, x, y, weights, null);

        if (isOffset) {
          //TODO: Subsample offset
          //glmnet.setOffset
        }

        outlist.add(glmnet_i.fit(sample.x, sample.y, sample.w));
      }
    }

    //TODO: tener en cuenta el resto de familias

    //model.measures == csvstuff
    Measures measures = Lognet.evaluate(outlist, lambdas, x, y, weights, foldid, measureType);
//      lambda = glmnet.object$lambda


    List<Double> cvm  = measures.cvm;
    List<Double> cvsd = measures.cvsd;
    lambda = lambdas.subList(0, cvsd.size());
    //TODO
//    nas = is.na(cvsd)
//    if (any(nas)) {
//      nz = nz[!nas]
//    }

//    out = list(lambda = lambda, cvm = cvm, cvsd = cvsd, cvup = cvm +
//        cvsd, cvlo = cvm - cvsd, nzero = nz, name = cvname, glmnet.model = glmnet.object)
//
//    if (keep) {
//      out = c(out, list(model.preval = cvstuff$model.preval, foldid = foldid))
//    }

    Pair<Double, Double> lamin;

    if (measures.type ==  MeasureType.AUC) {
      lamin = getMin(lambda, cvm.stream().map(d -> -d).collect(Collectors.toList()), cvsd);
    } else {
      lamin = getMin(lambda, cvm, cvsd);
    }

    double lambdaMin = lamin.getFirst();
    double lambda1se = lamin.getSecond();

    return new Model(lambdaMin, lambda1se, glmnetObject, measures);
  }

  private Pair<Double, Double> getMin(List<Double> lambda, List<Double> cvm, List<Double> cvsd) {
    final double cvmin = Collections.min(cvm);

    double lambdaMin = 0;
    double lambda1se = 0;
    double semin = -1;

    for (int i = 0; i < lambda.size(); ++i) {
      if (cvm.get(i) <= cvmin && lambda.get(i) > lambdaMin) {
        lambdaMin = lambda.get(i);
        semin = cvm.get(i) + cvsd.get(i);
      }
    }

    lambda1se = lambdaMin;

    for (int i = 0; i < lambda.size(); ++i) {
      if (cvm.get(i) <= semin && lambda.get(i) > lambda1se) {
        lambda1se = lambda.get(i);
      }
    }

    return new Pair<>(lambdaMin, lambda1se);
  }
}

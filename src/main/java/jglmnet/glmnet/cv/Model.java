package jglmnet.glmnet.cv;

import jglmnet.glmnet.ClassificationModel;
import jglmnet.glmnet.ClassificationModelSet;

/**
 * @author Jorge Peña
 */
public class Model {

  final public double lambdaMin;
  final public double lambda1se;
  private ClassificationModelSet models;
  private Measures measures;

  Model(double lambdaMin, double lambda1se, ClassificationModelSet models, Measures measures) {
    this.lambdaMin = lambdaMin;
    this.lambda1se = lambda1se;
    this.models = models;
    this.measures = measures;
  }

  public ClassificationModel coef(double s) throws Exception {
    return models.coef(s);
  }
}

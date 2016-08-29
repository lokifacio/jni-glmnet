package jglmnet.glmnet.cv;

import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

/**
 * @author Jorge Peña
 */
public class Sample {
  DenseDoubleMatrix2D x;
  DenseDoubleMatrix1D y;
  DenseDoubleMatrix1D w;
  DenseDoubleMatrix1D o;
}

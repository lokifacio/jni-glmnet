package jglmnet.glmnet.cv;

import cern.colt.matrix.tdouble.impl.DenseColumnDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tint.impl.DenseIntMatrix1D;

/**
 * @author Jorge Peña
 */
public class Sample {
  DenseIntMatrix1D pos;
  DenseColumnDoubleMatrix2D x;
  DenseDoubleMatrix1D y;
  DenseDoubleMatrix1D w;
  DenseDoubleMatrix1D o;
}

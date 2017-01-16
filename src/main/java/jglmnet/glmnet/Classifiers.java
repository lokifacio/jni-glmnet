package jglmnet.glmnet;

import cern.colt.matrix.tdouble.DoubleMatrix1D;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author Jorge Peña
 */
public class Classifiers {
  public static Map<Double, Long> getClassCount(DoubleMatrix1D y) {
    return Arrays.stream(y.toArray())
        .mapToObj(d -> d)
        .collect(Collectors.groupingBy(d -> d, Collectors.counting()));
  }

  public static int getNumberOfClasses(DoubleMatrix1D y) {
    return getClassCount(y).size();
  }
}

package jglmnet.glmnet;

import java.util.List;

/**
 * @author Jorge Peña
 */
public class Utils {
  public static int[] toIntArray(List<Integer> list) {
    return list.stream().mapToInt(Integer::intValue).toArray();
  }

  public static double[] toDoubleArray(List<Double> list) {
    return list.stream().mapToDouble(Double::doubleValue).toArray();
  }
}

package jglmnet.glmnet.cv;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Jorge Peña
 */
public class Measures implements Serializable {
  List<Double> cvm = new ArrayList<>();
  List<Double> cvsd = new ArrayList<>();
  MeasureType type;
}

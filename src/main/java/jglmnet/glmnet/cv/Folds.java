package jglmnet.glmnet.cv;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseColumnDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tint.impl.DenseIntMatrix1D;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Predicate;

/**
 * @author Jorge Peña
 */
class Folds {

  static Sample trainSamples(final List<Integer> foldid,
                                    final int fold,
                                    final DoubleMatrix2D x,
                                    final DoubleMatrix1D y,
                                    final DoubleMatrix1D weights,
                                    final DoubleMatrix1D offset) {
    return samples(foldid, x, y, weights, offset, sampleFold -> sampleFold != fold + 1);
  }

  static Sample testSamples(final List<Integer> foldid,
                                    final int fold,
                                    final DoubleMatrix2D x,
                                    final DoubleMatrix1D y,
                                    final DoubleMatrix1D weights,
                                    final DoubleMatrix1D offset) {
    return samples(foldid, x, y, weights, offset, sampleFold -> sampleFold == fold + 1);
  }

  private static Sample samples(final List<Integer> foldid,
                                final DoubleMatrix2D x,
                                final DoubleMatrix1D y,
                                final DoubleMatrix1D weights,
                                final DoubleMatrix1D offset,
                                final Predicate<Integer> cond) {

    int numSamples = (int) foldid.stream().filter(cond).count();

    Sample sample = new Sample();
    sample.pos = new DenseIntMatrix1D(numSamples);
    sample.x   = new DenseColumnDoubleMatrix2D(numSamples, x.columns());
    sample.y   = new DenseDoubleMatrix1D(numSamples);
    sample.w   = new DenseDoubleMatrix1D(numSamples);
    sample.o   = offset != null?new DenseDoubleMatrix1D(numSamples):null;

    int i_sub = 0;
    for (int i = 0; i < y.size(); ++i) {
      if (cond.test(foldid.get(i))) {
        sample.pos.set(i_sub, i);
        for (int j = 0; j < x.columns(); ++j) {
          sample.x.set(i_sub, j, x.get(i, j));
        }
        sample.y.set(i_sub, y.get(i));
        sample.w.set(i_sub, weights.get(i));
        if (offset != null) {
          sample.o.set(i_sub, offset.get(i));
        }

        i_sub++;
      }
    }

    return sample;
  }

  static int numFolds(List<Integer> foldid) {
    return Collections.max(foldid);
  }

  static List<Integer> generateFoldIds(final int numFolds, final int numSamples) {
    List<Integer> foldIds = new ArrayList<>(numSamples);

    int fold = 0;
    for (int i = 0; i < numSamples; ++i) {
      foldIds.add(fold + 1);
      fold = (fold + 1) % numFolds;
    }
    Collections.shuffle(foldIds);

    return foldIds;
  }
}

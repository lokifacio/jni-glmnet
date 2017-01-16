package jglmnet.glmnet;

import cz.adamh.utils.NativeUtils;

import java.io.IOException;

/**
 * Low-level bindings to glmnet functions.
 *
 * These closely mirror the FORTRAN functions of the same names, and follow FORTRAN conventions
 * (e.g. column-major representation for 2d arrays).  If in doubt, use the *Learner classes
 * instead.
 *
 * @author Thomas Down
 * @author Jorge Peña
 */

public class Fortran {
  static {
    try {
      NativeUtils.loadLibraryFromJar("/libgfortran.so.3");
      NativeUtils.loadLibraryFromJar("/" + System.mapLibraryName("glmnet"));
    } catch (IOException e) {
      e.printStackTrace(); // This is probably not the best way to handle exception :-)
    }
  }


  public native int elnet(
      int covUpdating,
      double alpha,
      double[] y,
      double[] w,
      double[] x,
      int[] mFlags,
      double[] penalties,
      int maxFinal,
      int maxPath,
      int numLambdas,
      double lambdaMinRatio,
      double[] userLambdas,
      double convThreshold,
      int standardize,
      int maxit,
      int[] outNumFits,
      double[] outIntercepts,
      double[] outCoeffs,
      int[] outCoeffPtrs,
      int [] outCoeffCnts,
      double[] outRsq,
      double[] outLambdas,
      int[] outNumPasses);

  /*
c call spelnet(ka,parm,no,ni,x,ix,jx,y,w,jd,vp,ne,nx,nlam,flmin,ulam,thr,
c             isd,maxit,lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
  */
  public native int spelnet(
      int covUpdating,
      double alpha,
      double[] y,
      double[] w,
      double[] xx,
      int[] xi,
      int [] xp,
      int[] mFlags,
      double[] penalties,
      int maxFinal,
      int maxPath,
      int numLambdas,
      double lambdaMinRatio,
      double[] userLambdas,
      double convThreshold,
      int standardize,
      int maxit,
      int[] outNumFits,
      double[] outIntercepts,
      double[] outCoeffs,
      int[] outCoeffPtrs,
      int [] outCoeffCnts,
      double[] outRsq,
      double[] outLambdas,
      int[] outNumPasses);

  /*
  c lognet (parm,no,ni,nc,x,y,o,jd,vp,ne,nx,nlam,flmin,ulam,thr,isd,
  c              maxit,kopt,lmu,a0,ca,ia,nin,dev0,fdev,alm,nlp,jerr)
  */
  public static native int lognet(
      double alpha,
      int nc,
      double[] y,
      double[] offsets,
      double[] x,
      int[] mFlags,
      double[] penalties,
      double[] coeffLimits,
      int maxFinal,
      int maxPath,
      int numLambdas,
      double lambdaMinRatio,
      double[] userLambdas,
      double convThreshold,
      int standardize,
      int intercept,
      int maxit,
      int kopt,
      int[] outNumFits,
      double[] outIntercepts,
      double[] outCoeffs,
      int[] outCoeffPtrs,
      int [] outCoeffCnts,
      double[] dev0,
      double[] fdev,
      double[] outLambdas,
      int[] outNumPasses);


  /*
  c call splognet (parm,no,ni,nc,x,ix,jx,y,o,jd,vp,ne,nx,nlam,flmin,
  c             ulam,thr,isd,maxit,kopt,lmu,a0,ca,ia,nin,dev0,fdev,alm,nlp,jerr)
  c
  */
  public native int splognet(
      double alpha,
      int nc,
      double[] y,
      double[] offsets,
      double[] xx,
      int[] xi,
      int [] xp,
      int[] mFlags,
      double[] penalties,
      int maxFinal,
      int maxPath,
      int numLambdas,
      double lambdaMinRatio,
      double[] userLambdas,
      double convThreshold,
      int standardize,
      int maxit,
      int kopt,
      int[] outNumFits,
      double[] outIntercepts,
      double[] outCoeffs,
      int[] outCoeffPtrs,
      int [] outCoeffCnts,
      double[] dev0,
      double[] fdev,
      double[] outLambdas,
      int[] outNumPasses);

  /*
  c fishhnet(parm,no,ni,x,y,o,w,jd,vp,ne,nx,nlam,flmin,ulam,thr,
  c               isd,maxit,lmu,a0,ca,ia,nin,dev0,fdev,alm,nlp,jerr)
  */

  /**
   *
   * @param alpha penalty member index [0,1]. 0: ridge, 1: lasso
   * @param y observation response counts
   * @param x
   * @param offsets observation off-sets
   * @param weights observation weights
   * @param mFlags predictor variable deletion flag
   *               jd(1) = 0  => use all variables
   *               jd(1) != 0 => do not use variables jd(2)...jd(jd(1)+1)
   * @param penalties relative penalties for each predictor variable
   *                  vp(j) = 0 => jth variable unpenalized
   * @param coeffLimits interval constraints on coefficient values
   *                    cl(1,j) = lower bound for jth coefficient value (<= 0.0)
   *                    cl(2,j) = upper bound for jth coefficient value (>= 0.0)
   * @param maxFinal maximum number of variables allowed to enter largest model (stopping criterion)
   * @param maxPath maximum number of variables allowed to enter all models
   *                along path (memory allocation, maxPath > maxFinal)
   * @param numLambdas (maximum) number of lamda values
   * @param lambdaMinRatio
   * @param userLambdas
   * @param convThreshold
   * @param standardize
   * @param intercept
   * @param maxit
   * @param outNumFits
   * @param outIntercepts
   * @param outCoeffs
   * @param outCoeffPtrs
   * @param outCoeffCnts
   * @param dev0
   * @param fdev
   * @param outLambdas
   * @param outNumPasses
   * @return error code
   */
  public static native int fishnet(
      double alpha,
      double[] y,
      double[] x,
      double[] offsets,
      double[] weights,
      int[] mFlags,
      double[] penalties,
      double[] coeffLimits,
      int maxFinal,
      int maxPath,
      int numLambdas,
      double lambdaMinRatio,
      double[] userLambdas,
      double convThreshold,
      int standardize,
      int intercept,
      int maxit,
      int[] outNumFits,
      double[] outIntercepts,
      double[] outCoeffs,
      int[] outCoeffPtrs,
      int [] outCoeffCnts,
      double[] dev0,
      double[] fdev,
      double[] outLambdas,
      int[] outNumPasses);
}
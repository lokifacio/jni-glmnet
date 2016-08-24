package jglmnet.glmnet;

/**
 * Error occurrent in Fortran learning routines.
 *
 * @author Thomas Down
 */

public class LearnerException extends RuntimeException {
    public LearnerException(int err) {
	super(String.format("Error in Fortran (err=%d)", err));
    }
}
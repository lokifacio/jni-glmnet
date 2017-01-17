# JNI-GLMneet

This is a fork on the JVM wrapper for glment by Thomas Down, which aims to mimic R's glmnet APIs.

Current version is developed using glmnet5.f90 which adds support for poisson families

For more information on glmnet, visit this [link](http://www.stanford.edu/~hastie/Papers/glmnet.pdf)

## Supported features:

Currently jglmnet supports and has been tested to output the same results as R glmnent 3.3.2 for the following features:

- GLMs:

| Family | Weights | Offsets |
| :----: | :-----: | :-----: |
| Binomial | X | X |
| Poisson  | X | X |

- Cross validation
  - Custom fold vector

- Parallel support

Pending features:
- Sparse matrix support

## Build libglmnet.so

```sh
$ gcc -I${JAVA_HOME}/../include -I${JAVA_HOME}/../include/linux -c glmnet.c -fPIC
$ gcc -c glmnet5.f90 -fdefault-real-8 -ffixed-form -fPIC
$ gcc -shared -o libglmnet.so glmnet.o glmnet5.o -lm -lgfortran -fPic
```

## Debug

```
java -agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=5005 -classpath glmnet-1.0.1-SNAPSHOT-jar-with-dependencies.jar jglmnet.glmnet.cv.PoissonTest
```
PHP_ARG_ENABLE(racplusplus, whether to enable racplusplus support,
[  --enable-racplusplus          Enable the racplusplus PHP extension], no)

PHP_ARG_WITH(racplusplus-eigen, for Eigen include directory,
[  --with-racplusplus-eigen[=DIR]  Path to Eigen3 headers],
[ PHP_RACPLUSPLUS_EIGEN="$withval" ],
[ PHP_RACPLUSPLUS_EIGEN="" ])

if test "$PHP_RACPLUSPLUS" != "no"; then
  PHP_REQUIRE_CXX()
  #CPPFLAGS="$CPPFLAGS -DRACPP_BUILDING_LIB_ONLY=1 -O1 -g -fsanitize=address -fno-omit-frame-pointer"
  #CXXFLAGS="$CXXFLAGS -std=gnu++17 -fopenmp -O1 -g -fsanitize=address -fno-omit-frame-pointer"
  #DFLAGS="-fsanitize=address"

  CPPFLAGS="$CPPFLAGS -DRACPP_BUILDING_LIB_ONLY=1"
  CXXFLAGS="$CXXFLAGS -std=gnu++17 -fopenmp"

  AC_DEFINE([RACPP_BUILDING_LIB_ONLY], [1], [Build the RAC++ core without pybind11])

  PHP_ADD_INCLUDE([$ext_srcdir/../src/racplusplus])

  if test -n "$PHP_RACPLUSPLUS_EIGEN" && test "$PHP_RACPLUSPLUS_EIGEN" != "no"; then
    PHP_ADD_INCLUDE([$PHP_RACPLUSPLUS_EIGEN])
  fi

  if test -n "$EIGEN3_INCLUDE_DIR"; then
    PHP_ADD_INCLUDE([$EIGEN3_INCLUDE_DIR])
  fi

  dnl ---- link-time libraries for this extension ----
  PHP_ADD_LIBRARY(stdc++, 1, RACPLUSPLUS_SHARED_LIBADD)
  PHP_ADD_LIBRARY(gomp, 1, RACPLUSPLUS_SHARED_LIBADD)
  PHP_SUBST(RACPLUSPLUS_SHARED_LIBADD)

  dnl ---- build the extension, with OpenMP flags for compilation ----
  PHP_NEW_EXTENSION(
    racplusplus,
    php_racplusplus.cpp ../src/racplusplus/_racplusplus.cpp,
    $ext_shared,
    RACPLUSPLUS_SHARED_LIBADD
  )
fi

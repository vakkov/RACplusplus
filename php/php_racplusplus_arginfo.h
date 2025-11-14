#ifndef PHP_RACPLUSPLUS_ARGINFO_H
#define PHP_RACPLUSPLUS_ARGINFO_H

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_racplusplus_rac, 0, 2, IS_ARRAY, 0)
    ZEND_ARG_ARRAY_INFO(0, points, 0)
    ZEND_ARG_TYPE_INFO(0, max_merge_distance, IS_DOUBLE, 0)
    ZEND_ARG_ARRAY_INFO(0, connectivity, 1)
    ZEND_ARG_TYPE_INFO(0, batch_size, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, no_processors, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, distance_metric, IS_STRING, 0)
ZEND_END_ARG_INFO()

#endif // PHP_RACPLUSPLUS_ARGINFO_H

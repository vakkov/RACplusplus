extern "C" {
#include "php.h"
#include "ext/standard/info.h"
#include "zend_exceptions.h"
}

#include <memory>
#include <string>
#include <vector>

#include "php_racplusplus.h"
#include "php_racplusplus_arginfo.h"
#include "../src/racplusplus/_racplusplus.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


namespace racplusplus_php {

typedef Eigen::Index index_t;

bool convert_points_array(zval* points, Eigen::MatrixXd& matrix, std::string& error) {
    if (Z_TYPE_P(points) != IS_ARRAY) {
        error = "points must be a numeric array of arrays";
        return false;
    }

    HashTable* outer = Z_ARRVAL_P(points);
    const index_t rows = static_cast<index_t>(zend_hash_num_elements(outer));
    if (rows == 0) {
        error = "points must contain at least one row";
        return false;
    }

    index_t cols = -1;
    index_t row_index = 0;

    zval* row_zv;
    ZEND_HASH_FOREACH_VAL(outer, row_zv) {
        if (Z_TYPE_P(row_zv) != IS_ARRAY) {
            error = "each row in points must itself be an array";
            return false;
        }

        HashTable* row_ht = Z_ARRVAL_P(row_zv);
        const index_t current_cols = static_cast<index_t>(zend_hash_num_elements(row_ht));
        if (current_cols == 0) {
            error = "points rows cannot be empty";
            return false;
        }

        if (cols == -1) {
            cols = current_cols;
            matrix.resize(rows, cols);
        } else if (current_cols != cols) {
            error = "each row in points must use the same dimensionality";
            return false;
        }

        index_t col_index = 0;
        zval* value;
        ZEND_HASH_FOREACH_VAL(row_ht, value) {
            matrix(row_index, col_index) = zval_get_double(value);
            ++col_index;
        } ZEND_HASH_FOREACH_END();

        ++row_index;
    } ZEND_HASH_FOREACH_END();

    return true;
}

bool convert_connectivity_matrix(
    zval* connectivity,
    index_t expected_size,
    std::unique_ptr<Eigen::SparseMatrix<bool>>& matrix,
    std::string& error) {

    if (connectivity == nullptr || Z_TYPE_P(connectivity) == IS_UNDEF || Z_TYPE_P(connectivity) == IS_NULL) {
        return true;
    }

    if (Z_TYPE_P(connectivity) == IS_FALSE) {
        return true;
    }

    if (Z_TYPE_P(connectivity) != IS_ARRAY) {
        error = "connectivity must be an NxN boolean array";
        return false;
    }

    HashTable* outer = Z_ARRVAL_P(connectivity);
    if (static_cast<index_t>(zend_hash_num_elements(outer)) != expected_size) {
        error = "connectivity must have the same number of rows as points";
        return false;
    }

    std::vector<Eigen::Triplet<bool>> triplets;
    triplets.reserve(static_cast<size_t>(expected_size));

    index_t row_index = 0;
    zval* row_zv;
    ZEND_HASH_FOREACH_VAL(outer, row_zv) {
        if (Z_TYPE_P(row_zv) != IS_ARRAY) {
            error = "each row in connectivity must be an array";
            return false;
        }

        HashTable* row_ht = Z_ARRVAL_P(row_zv);
        if (static_cast<index_t>(zend_hash_num_elements(row_ht)) != expected_size) {
            error = "connectivity must be an NxN matrix";
            return false;
        }

        index_t col_index = 0;
        zval* value;
        ZEND_HASH_FOREACH_VAL(row_ht, value) {
            if (zend_is_true(value)) {
                triplets.emplace_back(row_index, col_index, true);
            }
            ++col_index;
        } ZEND_HASH_FOREACH_END();

        ++row_index;
    } ZEND_HASH_FOREACH_END();

    matrix = std::make_unique<Eigen::SparseMatrix<bool>>(expected_size, expected_size);
    matrix->setFromTriplets(triplets.begin(), triplets.end());
    return true;
}

} // namespace racplusplus_php

PHP_FUNCTION(racplusplus_rac) {
    zval* points_zv = nullptr;
    double max_merge_distance = 0.0;
    zval* connectivity_zv = nullptr;
    zend_long batch_size = 0;
    zend_long no_processors = 0;
    zend_string* distance_metric = nullptr;

    ZEND_PARSE_PARAMETERS_START(2, 6)
        Z_PARAM_ZVAL(points_zv)
        Z_PARAM_DOUBLE(max_merge_distance)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(connectivity_zv)
        Z_PARAM_LONG(batch_size)
        Z_PARAM_LONG(no_processors)
        Z_PARAM_STR_OR_NULL(distance_metric)
    ZEND_PARSE_PARAMETERS_END();

    Eigen::MatrixXd base_arr;
    std::string error;
    if (!racplusplus_php::convert_points_array(points_zv, base_arr, error)) {
        zend_throw_exception_ex(zend_ce_exception, 0, "%s", error.c_str());
        RETURN_THROWS();
    }

    std::unique_ptr<Eigen::SparseMatrix<bool>> connectivity_matrix;
    Eigen::SparseMatrix<bool>* connectivity_ptr = nullptr;
    if (connectivity_zv != nullptr && Z_TYPE_P(connectivity_zv) != IS_UNDEF && Z_TYPE_P(connectivity_zv) != IS_NULL) {
        if (!racplusplus_php::convert_connectivity_matrix(connectivity_zv, base_arr.rows(), connectivity_matrix, error)) {
            zend_throw_exception_ex(zend_ce_exception, 0, "%s", error.c_str());
            RETURN_THROWS();
        }
        connectivity_ptr = connectivity_matrix.get();
    }

    std::string metric = "euclidean";
    if (distance_metric != nullptr && ZSTR_LEN(distance_metric) > 0) {
        metric.assign(ZSTR_VAL(distance_metric), ZSTR_LEN(distance_metric));
    }

    std::vector<int> labels = RAC(
        base_arr,
        max_merge_distance,
        connectivity_ptr,
        static_cast<int>(batch_size),
        static_cast<int>(no_processors),
        metric);

    array_init_size(return_value, labels.size());
    for (const int label : labels) {
        add_next_index_long(return_value, label);
    }
}

PHP_MINIT_FUNCTION(racplusplus) {
    REGISTER_STRING_CONSTANT("RACPLUSPLUS_VERSION", PHP_RACPLUSPLUS_VERSION, CONST_CS | CONST_PERSISTENT);
    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(racplusplus) {
    UPDATE_NEIGHBOR_DURATIONS.clear();
    UPDATE_NN_DURATIONS.clear();
    COSINE_DURATIONS.clear();
    INDICES_DURATIONS.clear();
    MERGE_DURATIONS.clear();
    MISC_MERGE_DURATIONS.clear();
    INITIAL_NEIGHBOR_DURATIONS.clear();
    HASH_DURATIONS.clear();
    UPDATE_PERCENTAGES.clear();
    return SUCCESS;
}

PHP_MINFO_FUNCTION(racplusplus) {
    php_info_print_table_start();
    php_info_print_table_row(2, "racplusplus support", "enabled");
    php_info_print_table_row(2, "Version", PHP_RACPLUSPLUS_VERSION);
    php_info_print_table_end();
}

static const zend_function_entry racplusplus_functions[] = {
    PHP_FE(racplusplus_rac, arginfo_racplusplus_rac)
    PHP_FE_END
};

zend_module_entry racplusplus_module_entry = {
    STANDARD_MODULE_HEADER,
    "racplusplus",
    racplusplus_functions,
    PHP_MINIT(racplusplus),
    PHP_MSHUTDOWN(racplusplus),
    nullptr,
    nullptr,
    PHP_MINFO(racplusplus),
    PHP_RACPLUSPLUS_VERSION,
    STANDARD_MODULE_PROPERTIES
};

#ifdef COMPILE_DL_RACPLUSPLUS
extern "C" {
ZEND_GET_MODULE(racplusplus)
}
#endif

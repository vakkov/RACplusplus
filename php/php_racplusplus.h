#ifndef PHP_RACPLUSPLUS_H
#define PHP_RACPLUSPLUS_H

extern "C" {
#include "php.h"
}

#define PHP_RACPLUSPLUS_VERSION "0.9.0-dev"

extern zend_module_entry racplusplus_module_entry;
#define phpext_racplusplus_ptr &racplusplus_module_entry

PHP_MINIT_FUNCTION(racplusplus);
PHP_MINFO_FUNCTION(racplusplus);

PHP_FUNCTION(racplusplus_rac);

#endif // PHP_RACPLUSPLUS_H

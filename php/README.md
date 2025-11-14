# RAC++ PHP extension (experimental)

This directory contains a minimal PHP extension that exposes the `rac()` clustering routine implemented in `src/racplusplus`. The goal is parity with the Python bindings, but the surface area is intentionally small while the API stabilises.

## Requirements

- PHP 8.1+ with the development headers (`php-dev`, `php-devel`, etc.)
- A working C++17 toolchain
- Eigen 3.4 headers available on your include path (set `EIGEN3_INCLUDE_DIR` or pass `--with-racplusplus-eigen=/path/to/eigen3`)
- The existing RAC++ dependencies already used for the Python extension (OpenMP optional)

## Building

```
cd php
phpize
./configure --enable-racplusplus --with-racplusplus-eigen=/usr/include/eigen3
make
make test   # optional; currently no PHPTs are defined
sudo make install
```

`phpize` will generate the build system for your locally installed PHP. If Eigen is installed in a non-standard location, set `EIGEN3_INCLUDE_DIR=/path/to/eigen3` before running `./configure` or use the `--with-racplusplus-eigen` switch shown above.

After installation, enable the extension by adding the resulting `racplusplus.so` to your `php.ini` (or use `extension=/full/path/racplusplus.so`).

## Usage

```php
<?php
$points = [
    [0.1, 0.2, 0.3],
    [0.05, 0.2, 0.28],
    [0.9, 0.8, 0.75],
];

$labels = racplusplus_rac($points, 0.24, null, 500, 8, 'cosine');
print_r($labels);
```

- `$points`: array of N rows, each row an array of `float`/`int` values.
- `$max_merge_distance`: identical tuning parameter to the Python API.
- `$connectivity`: optional NxN boolean matrix. Use `null` (default) to fall back to full connectivity.
- `$batch_size`, `$no_processors`, `$distance_metric`: match the Python binding semantics.

The function returns a simple PHP array of cluster ids ordered like the input rows.

## Notes

- The PHP module reuses the same C++ translation unit as the Python extension by defining `RACPP_BUILDING_LIB_ONLY`, so the heavy lifting stays in one place.
- Only the `racplusplus_rac()` entry point is exposed today. The distance helper methods can be added later following the same pattern once there is demand.
- Connectivity conversion currently expects a dense PHP array; if you frequently work with sparse connectivities, consider extending `php/php_racplusplus.cpp` with a more compact format.

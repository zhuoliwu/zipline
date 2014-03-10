Release Notes
=============

List of changes between Zipline releases.

### Distribution Links

- Source code: https://github.com/quantopian/zipline
- Downloads: https://pypi.python.org/pypi/zipline

## Zipline 0.6.1

*Release date:* (not yet released)

### Enhancements

- Performance enhancement, removed `alias_dt` transform.
  (a203f6963544170d218572343f1dab5e57d8a2cf)

### Bug Fixes

- Fixed floating point error in `order()`. #280
- Fixed cost basis calculation, order direction no longer affects cost basis. #278
- Fixed max drawdown calculation, max drawdown was not using a value representatitve
  of the algorithm's total return at the given time.
  (6cdd5ddb1026658893e606496f209c06c6f3f068)

### Release/Build

- Added a conda channel for precompiled builds.

### Maintenance


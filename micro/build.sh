#!/bin/zsh
set -e
cd "$(dirname $0)"
for k in kernels qgemm; do
  xcrun metal -std=metal4.0 -O3 -c $k.metal -o $k.air
  xcrun metallib $k.air -o $k.metallib
done
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation micro.mm -o micro
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation qtest.mm -o qtest
xcrun metal -std=metal4.0 -O3 -c w4a8.metal -o w4a8.air && xcrun metallib w4a8.air -o w4a8.metallib
clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation w4a8.mm -o w4a8

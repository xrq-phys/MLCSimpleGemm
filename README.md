Calling Apple's ML Compute SGEMM from C.

To compile:

```bash
# Compile Swift
swiftc -emit-library -c MLCSimpleGemm.swift -o MLCSimpleGemm.o
# Link to dynamic library.
# Use a intermediate layer to fix symbol.
clang -shared csymbol.c MLCSimpleGemm.o -L/usr/lib/swift -lswiftCore -o MLCSimpleGemm.dylib

# With the symbol fixed, it's now possible to directly link from a usual C program.
clang example_sgemm_call.c MLCSimpleGemm.dylib -o example_sgemm_call.x
```


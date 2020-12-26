Calling Apple's ML Compute SGEMM from C.

To compile:

```bash
clang -c example_sgemm_call.c -o example_sgemm_call.o
swiftc -emit-library -c MLCSimpleGemm.swift -o MLCSimpleGemm.o
swiftc example_sgemm_call.o MLCSimpleGemm.o -o example_sgemm_call.x
```


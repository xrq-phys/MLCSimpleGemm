using Libdl

if ! (@isdefined libmlc)
    libmlc = dlopen("MLCSimpleGemm.dylib")
end

"Compute SGEMM with ML Compute."
mmul_simple!(tA::Bool,
             tB::Bool,
             α::Float32,
             A::Matrix{Float32},
             B::Matrix{Float32},
             C::Matrix{Float32}) = begin
    if !tA
        m, k = size(A)
    else
        k, m = size(A)
    end
    if !tB
        k_,n = size(B)
    else
        n, k_= size(B)
    end
    m_, n_ = size(C)
    (k == k_ && m == m_ && n == n_) ||
        throw(DimensionMismatch("Array dimension mismatch."))

    # ML Compute is row-major. Exchange A and B.
    ccall(dlsym(libmlc, "mlcsgemm_simple"),
          Cvoid,
          (Bool, Bool,
           Cint, Cint, Cint,
           Float32,
           Ptr{Float32},
           Ptr{Float32},
           Ptr{Float32}),
          tB, tA,
          n, m, k,
          α,
          B, A, C);
    C
end


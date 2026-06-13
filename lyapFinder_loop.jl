using LinearAlgebra
using Random
using CSV
using DataFrames
using Statistics
using Printf

BLAS.set_num_threads(1)

# ============================================================
# Transfer matrices  (unchanged from canonical)
# ============================================================

function B1(ex::Float64)
    s = sqrt(1.0 + ex)
    t = 1.0 / sqrt(1.0 + 1.0 / ex)
    return s * [1.0 t; t 1.0]
end

function B2(ex::Float64)
    s = sqrt(1.0 + 1.0 / ex)
    t = 1.0 / sqrt(1.0 + ex)
    return s * [1.0 t; t 1.0]
end

# ============================================================
# S-to-T conversion  (shared helper)
#
# Given a 2×2 scattering matrix S = [[s11, s12],[s21, s22]]
# the transfer matrix is:
#   T = [[s11 - s12/s22*s21,  s12/s22],
#        [      -s21/s22,     1/s22  ]]
# ============================================================

@inline function S_to_T(S::Matrix{ComplexF64})::Matrix{ComplexF64}
    s11 = S[1,1]; s12 = S[1,2]
    s21 = S[2,1]; s22 = S[2,2]
    inv22 = 1.0 / s22
    return [s11 - s12 * inv22 * s21    s12 * inv22;
                       -inv22 * s21         inv22 ]
end

# ============================================================
# cR computation from the Mathematica formula (Out[9])
#
# Inputs: θ and four uniform phases ΨA, ΨB, ΨC, ΨD ∈ [0, 2π)
#
# The formula is  cR = |S̃̃[1,1]| = Numerator / Denominator
# where (transcribing Out[9] exactly):
#
# Let c = cos(θ), s = sin(θ), and abbreviate:
#   cosPA  = cos(ΨA),   sinPA  = sin(ΨA)
#   cosPAB = cos(ΨA+ΨB), etc.
#
# Num_re = -cos(ΨA) + c³ cos(ΨA+ΨB) - c³ cos(ΨC+ΨD)
#          + cos(ΨA+ΨC+ΨD) - cos(ΨA+ΨC) s + cos(ΨA+ΨD) s
#
# Num_im = -sin(ΨA) + c³ sin(ΨA+ΨB) - s sin(ΨA+ΨC)
#          + s sin(ΨA+ΨD) - c³ sin(ΨC+ΨD) + sin(ΨA+ΨC+ΨD)
#
# |Num|² = s⁴ * (Num_re² + Num_im²)
#
# Den_re1 = c³(-cos(ΨA)-cos(ΨB)) + c⁶ cos(ΨA+ΨB)
#           + c⁴ cos(ΨA+ΨB) s² - c² cos(ΨC+ΨD) s⁴
#           + (-1 - cos(ΨC) s³)(-1 + cos(ΨD) s³)
#           + s⁶ sin(ΨC) sin(ΨD)
#
# Den_re2 = c³(-sin(ΨA)-sin(ΨB)) + c⁶ sin(ΨA+ΨB)
#           + c⁴ s² sin(ΨA+ΨB) - s³(-1 + cos(ΨD) s³) sin(ΨC)
#           + sin(ΨD) s³(-1 - cos(ΨC) s³)
#           - c² s⁴ sin(ΨC+ΨD)
#
# |Den|² = Den_re1² + Den_re2²
#
# cR = s² * sqrt(Num_re² + Num_im²) / sqrt(Den_re1² + Den_re2²)
# ============================================================

@inline function compute_cR(theta::Float64, PsiA::Float64, PsiB::Float64,
                             PsiC::Float64, PsiD::Float64)::Float64
    c  = cos(theta)
    s  = sin(theta)
    c2 = c * c
    c3 = c2 * c
    c4 = c2 * c2
    c6 = c3 * c3
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    s6 = s3 * s3

    # Precompute trig combinations
    cosA    = cos(PsiA)
    sinA    = sin(PsiA)
    cosB    = cos(PsiB)
    sinB    = sin(PsiB)
    cosC    = cos(PsiC)
    sinC    = sin(PsiC)
    cosD    = cos(PsiD)
    sinD    = sin(PsiD)
    cosAB   = cos(PsiA + PsiB)
    sinAB   = sin(PsiA + PsiB)
    cosCD   = cos(PsiC + PsiD)
    sinCD   = sin(PsiC + PsiD)
    cosAC   = cos(PsiA + PsiC)
    sinAC   = sin(PsiA + PsiC)
    cosAD   = cos(PsiA + PsiD)
    sinAD   = sin(PsiA + PsiD)
    cosACD  = cos(PsiA + PsiC + PsiD)
    sinACD  = sin(PsiA + PsiC + PsiD)

    # Numerator real and imaginary parts (before the s² prefactor)
    Num_re = -cosA + c3*cosAB - c3*cosCD + cosACD - cosAC*s + cosAD*s
    Num_im = -sinA + c3*sinAB - s*sinAC   + s*sinAD  - c3*sinCD + sinACD

    num2 = s4 * (Num_re^2 + Num_im^2)

    # Denominator real part
    Den_re = (c3*(-cosA - cosB)
              + c6*cosAB
              + c4*cosAB*s2
              - c2*cosCD*s4
              + (-1.0 - cosC*s3)*(-1.0 + cosD*s3)
              + s6*sinC*sinD)

    # Denominator imaginary part
    Den_im = (c3*(-sinA - sinB)
              + c6*sinAB
              + c4*s2*sinAB
              - s3*(-1.0 + cosD*s3)*sinC
              + sinD*s3*(-1.0 - cosC*s3)
              - c2*s4*sinCD)

    den2 = Den_re^2 + Den_im^2

    # Safety clamp: den2 should never be zero for generic phases,
    # but guard against numerical noise producing cR slightly > 1
    cR_sq = num2 / den2
    return sqrt(clamp(cR_sq, 0.0, 1.0))
end

# ============================================================
# Loop transfer matrices
#
# S_loop = [[cR, -sR], [sR, cR]]   (real rotation matrix)
# where sR = sqrt(1 - cR²)
#
# loop_B1 uses the B1-type S-to-T conversion:
#   off-diagonal blocks carry the 1/s22 factor
# loop_B2 uses the B2-type S-to-T conversion.
#
# The distinction between B1 and B2 is purely which channel
# plays the role of s22 (i.e. which index is the "transmitted"
# direction).  In the existing code B1 ↔ B2 differ by swapping
# the indexing, which here corresponds to transposing S before
# applying S_to_T.
# ============================================================

function loop_B1(theta::Float64, PsiA::Float64, PsiB::Float64,
                 PsiC::Float64, PsiD::Float64)::Matrix{ComplexF64}
    cR = compute_cR(theta, PsiA, PsiB, PsiC, PsiD)
    sR = sqrt(1.0 - cR^2)
    S  = ComplexF64[cR  -sR;
                    sR   cR]
    return S_to_T(S)
end

function loop_B2(theta::Float64, PsiA::Float64, PsiB::Float64,
                 PsiC::Float64, PsiD::Float64)::Matrix{ComplexF64}
    cR = compute_cR(theta, PsiA, PsiB, PsiC, PsiD)
    sR = sqrt(1.0 - cR^2)
    # B2-type: transpose S (swap s12↔s21) before converting,
    # consistent with how the existing B2 relates to B1
    S  = ComplexF64[ cR  sR;
                    -sR  cR]
    return S_to_T(S)
end

# ============================================================
# In-place block matrix actions  (unchanged)
# ============================================================

function M1Action!(Z::Matrix{ComplexF64}, buf::Matrix{ComplexF64},
                   B::Matrix{ComplexF64}, M::Int)
    @inbounds for i in 1:M
        r1 = 2i - 1
        r2 = 2i
        for k in axes(Z, 2)
            z1 = Z[r1, k]
            z2 = Z[r2, k]
            buf[r1, k] = B[1,1]*z1 + B[1,2]*z2
            buf[r2, k] = B[2,1]*z1 + B[2,2]*z2
        end
    end
end

function M2Action!(Z::Matrix{ComplexF64}, buf::Matrix{ComplexF64},
                   B::Matrix{ComplexF64}, M::Int)
    n = 2 * M
    @inbounds for i in 1:M-1
        r1 = 2i
        r2 = 2i + 1
        for k in axes(Z, 2)
            z1 = Z[r1, k]
            z2 = Z[r2, k]
            buf[r1, k] = B[1,1]*z1 + B[1,2]*z2
            buf[r2, k] = B[2,1]*z1 + B[2,2]*z2
        end
    end
    @inbounds for k in axes(Z, 2)
        z1 = Z[n, k]
        z2 = Z[1,  k]
        buf[n, k] = B[1,1]*z1 + B[1,2]*z2
        buf[1, k] = B[2,1]*z1 + B[2,2]*z2
    end
end

# ============================================================
# TransfMAction! — extended to support loop insertions
#
# Extra keyword arguments (ignored when loop_prob == 0):
#   loop_prob :: Float64       — probability p of inserting a loop at
#                                each site per time step
#   theta     :: Float64       — scattering angle (atan(sqrt(ex)))
#   rng       :: AbstractRNG   — per-thread RNG for drawing Ψ phases
#   psi_buf   :: Vector{Float64}(undef,4)  — caller-allocated scratch
#   site_B1   :: Vector{Matrix{ComplexF64}}(undef,M)   — caller-allocated
#   site_B2   :: Vector{Matrix{ComplexF64}}(undef,M)   — caller-allocated
#
# When loop_prob == 0 the function is byte-for-byte equivalent to
# the canonical version (fast path, no extra work).
#
# Design: before touching Z we pre-select, for each strip site i,
# whether a loop fires and if so compute the replacement 2×2 matrix.
# This avoids any control flow inside the hot column loop and keeps
# the row-pair indexing identical to M1Action!/M2Action!.
#
# M1 pairs: (2i-1, 2i)   for i = 1..M          — use site_B1[i]
# M2 pairs: (2i,   2i+1) for i = 1..M-1,        — use site_B2[i]
#           (2M,   1)    for i = M  (wrap-around)
# ============================================================

function TransfMAction!(Z::Matrix{ComplexF64},
                        buf1::Matrix{ComplexF64},
                        buf2::Matrix{ComplexF64},
                        U1::AbstractVector{ComplexF64},
                        U2::AbstractVector{ComplexF64},
                        B1m::Matrix{ComplexF64},
                        B2m::Matrix{ComplexF64},
                        M::Int;
                        loop_prob::Float64 = 0.0,
                        theta::Float64     = 0.0,
                        rng::Union{Nothing, AbstractRNG} = nothing,
                        psi_buf::Vector{Float64} = Float64[],
                        site_B1::Vector{Matrix{ComplexF64}} = Matrix{ComplexF64}[],
                        site_B2::Vector{Matrix{ComplexF64}} = Matrix{ComplexF64}[])

    n = 2 * M

    # ── Phase multiply by U2 ──────────────────────────────────────────────
    @inbounds for k in axes(Z, 2)
        for r in 1:n
            buf1[r, k] = U2[r] * Z[r, k]
        end
    end

    # ── M2 (B2) action ────────────────────────────────────────────────────
    if loop_prob == 0.0
        M2Action!(buf1, buf2, B2m, M)
    else
        # Pre-select per-site matrix for all M pairs before the column loop.
        # site_B2[i] is the 2×2 block for M2 pair i.
        @inbounds for i in 1:M
            if rand(rng) < loop_prob
                rand!(rng, psi_buf)
                site_B2[i] = loop_B2(theta, 2π*psi_buf[1], 2π*psi_buf[2],
                                              2π*psi_buf[3], 2π*psi_buf[4])
            else
                site_B2[i] = B2m
            end
        end
        # Apply: pairs (2i, 2i+1) for i=1..M-1
        @inbounds for i in 1:M-1
            r1 = 2i
            r2 = 2i + 1
            B  = site_B2[i]
            for k in axes(buf1, 2)
                z1 = buf1[r1, k]
                z2 = buf1[r2, k]
                buf2[r1, k] = B[1,1]*z1 + B[1,2]*z2
                buf2[r2, k] = B[2,1]*z1 + B[2,2]*z2
            end
        end
        # Wrap-around pair (2M, 1) uses site_B2[M]
        @inbounds begin
            B  = site_B2[M]
            for k in axes(buf1, 2)
                z1 = buf1[2M, k]
                z2 = buf1[1,  k]
                buf2[2M, k] = B[1,1]*z1 + B[1,2]*z2
                buf2[1,  k] = B[2,1]*z1 + B[2,2]*z2
            end
        end
    end

    # ── Phase multiply by U1 ──────────────────────────────────────────────
    @inbounds for k in axes(Z, 2)
        for r in 1:n
            buf1[r, k] = U1[r] * buf2[r, k]
        end
    end

    # ── M1 (B1) action ────────────────────────────────────────────────────
    if loop_prob == 0.0
        M1Action!(buf1, Z, B1m, M)
    else
        # Pre-select per-site matrix for all M pairs.
        @inbounds for i in 1:M
            if rand(rng) < loop_prob
                rand!(rng, psi_buf)
                site_B1[i] = loop_B1(theta, 2π*psi_buf[1], 2π*psi_buf[2],
                                              2π*psi_buf[3], 2π*psi_buf[4])
            else
                site_B1[i] = B1m
            end
        end
        # Apply: pairs (2i-1, 2i) for i=1..M
        @inbounds for i in 1:M
            r1 = 2i - 1
            r2 = 2i
            B  = site_B1[i]
            for k in axes(buf1, 2)
                z1 = buf1[r1, k]
                z2 = buf1[r2, k]
                Z[r1, k] = B[1,1]*z1 + B[1,2]*z2
                Z[r2, k] = B[2,1]*z1 + B[2,2]*z2
            end
        end
    end
end

# ============================================================
# Lyapunov exponent helper  (unchanged)
# ============================================================

function localizationγ(list::Vector{Float64})
    neg = filter(x -> x < 0, list)
    if isempty(neg)
        println("Warning: list is empty")
        return NaN
    end
    return -maximum(neg)
end

# ============================================================
# Gram-Schmidt orthogonalization  (zero-allocation, from Deven)
# ============================================================

function gram_schmidt!(Z::Matrix{ComplexF64}, γlist::Vector{Float64},
                       Lcorr::Float64)
    n = size(Z, 1)
    @inbounds for k in 1:n
        # Normalize column k
        col_norm = 0.0
        for row in 1:n; col_norm += abs2(Z[row, k]); end
        col_norm = sqrt(col_norm)
        γlist[k] += log(col_norm) / Lcorr
        for row in 1:n; Z[row, k] /= col_norm; end
        # Subtract projection of column k from all subsequent columns
        for next_col in (k+1):n
            dot_prod = ComplexF64(0.0)
            for row in 1:n
                dot_prod += conj(Z[row, k]) * Z[row, next_col]
            end
            for row in 1:n
                Z[row, next_col] -= dot_prod * Z[row, k]
            end
        end
    end
end

# ============================================================
# Single realization — loop-extended version
#
# loop_prob = 0.0  →  identical to canonical (p=0 validation)
# loop_prob > 0.0  →  each of the M sites is a candidate for
#                     loop insertion with probability loop_prob
# ============================================================

function lyapfinder_single(xlist::Vector{Float64}, M::Int, L::Int, w::Int,
                           rng::AbstractRNG;
                           loop_prob::Float64 = 0.0)
    s     = ceil(Int, L / w)
    Lcorr = Float64(w * s)
    n     = 2 * M
    xN    = length(xlist)

    results = Vector{Float64}(undef, xN)

    U1        = Vector{ComplexF64}(undef, n)
    U2        = Vector{ComplexF64}(undef, n)
    row_theta = Vector{Float64}(undef, n)

    # Pre-allocate scratch for loop insertion (reused across all steps).
    # When loop_prob == 0 these are never accessed.
    psi_buf = Vector{Float64}(undef, 4)
    site_B1 = [Matrix{ComplexF64}(undef, 2, 2) for _ in 1:M]
    site_B2 = [Matrix{ComplexF64}(undef, 2, 2) for _ in 1:M]

    for c in 1:xN
        ex    = exp(2.0 * xlist[c])
        theta = atan(sqrt(ex))          # tan(α) = e^x  →  α = atan(e^x)
        B1m   = ComplexF64.(B1(ex))
        B2m   = ComplexF64.(B2(ex))
        Z     = Matrix{ComplexF64}(I, n, n)
        buf1  = Matrix{ComplexF64}(undef, n, n)
        buf2  = Matrix{ComplexF64}(undef, n, n)
        γlist = zeros(Float64, n)

        rng_copy = copy(rng)

        for i in 1:s
            for j in 1:w
                rand!(rng_copy, row_theta)
                @inbounds for k in 1:n
                    U1[k] = cis(2π * row_theta[k])
                end
                rand!(rng_copy, row_theta)
                @inbounds for k in 1:n
                    U2[k] = cis(2π * row_theta[k])
                end
                TransfMAction!(Z, buf1, buf2, U1, U2, B1m, B2m, M;
                                loop_prob = loop_prob,
                                theta     = theta,
                                rng       = rng_copy,
                                psi_buf   = psi_buf,
                                site_B1   = site_B1,
                                site_B2   = site_B2)
            end
            gram_schmidt!(Z, γlist, Lcorr)
        end

        results[c] = M * localizationγ(γlist)
    end

    return results
end

# ============================================================
# Multi-realization — loop-extended version
# ============================================================

function lyapfinder_multi(xlist::Vector{Float64}, M::Int, L::Int, w::Int,
                          n_real::Int, master_seed::Int;
                          loop_prob::Float64 = 0.0)

    xN = length(xlist)
    γ_accum = Matrix{Float64}(undef, n_real, xN)

    master_rng = MersenneTwister(master_seed)
    seeds = rand(master_rng, UInt64, n_real)

    println("Starting $n_real realizations across $(Threads.nthreads()) threads...")
    println("Loop insertion probability p = $loop_prob")

    Threads.@threads for r in 1:n_real
        rng = MersenneTwister(seeds[r])
        γ_accum[r, :] = lyapfinder_single(xlist, M, L, w, rng;
                                           loop_prob = loop_prob)
        if r % max(1, n_real ÷ 10) == 0
            @info "Completed $r / $n_real realizations"
        end
    end

    results = Vector{Tuple{Float64, Float64, Float64}}(undef, xN)
    for c in 1:xN
        γ_vals = γ_accum[:, c]
        results[c] = (xlist[c], mean(γ_vals), std(γ_vals))
    end

    return results
end

# ============================================================
# Helpers
# ============================================================

function print_results(results::Vector{Tuple{Float64, Float64, Float64}})
    println("\nx\t\tmean_γ\t\tstd_γ")
    println("-"^50)
    for (x, μ, σ) in results
        @printf("%.4f\t\t%.6f\t%.6f\n", x, μ, σ)
    end
end

function save_results(results::Vector{Tuple{Float64, Float64, Float64}},
                      filepath::String)
    df = DataFrame(
        x      = [r[1] for r in results],
        mean_γ = [r[2] for r in results],
        std_γ  = [r[3] for r in results]
    )
    CSV.write(filepath, df)
    println("Results saved to $filepath")
end

# ============================================================
# Validation helpers
# ============================================================

"""
    validate_unitarity(theta; N=1000)

Draw N random sets of (ΨA,ΨB,ΨC,ΨD) and check that S_loop = [[cR,-sR],[sR,cR]]
is unitary (trivially true by construction) and that cR ∈ [0,1].
Returns (min_cR, max_cR, all_in_range).
"""
function validate_unitarity(theta::Float64; N::Int=1000)
    rng = MersenneTwister(0)
    cRs = [compute_cR(theta, 2π*rand(rng), 2π*rand(rng),
                      2π*rand(rng), 2π*rand(rng)) for _ in 1:N]
    min_cR = minimum(cRs)
    max_cR = maximum(cRs)
    return min_cR, max_cR, (min_cR >= 0.0 && max_cR <= 1.0)
end

"""
    validate_zero_phases(theta)

Check the bare limit: all Ψ = 0.  Analytically S_eff = diag(1, -1)
so cR should equal 1 (rotation angle = 0 or π).
"""
function validate_zero_phases(theta::Float64)
    cR = compute_cR(theta, 0.0, 0.0, 0.0, 0.0)
    println("Zero-phase cR at theta=$(round(theta, digits=4)): $cR  (expected 1.0)")
    return cR
end

MList = [20]
pList = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

for p in pList
    for M in MList
        results = lyapfinder_multi(
        collect(0.0:0.01:0.2),   # xlist
        M,                        # M
        1000000,                  # L
        20,                       # w
        768,                      # n_real
        42,                       # seed
        loop_prob=p
        )
        print_results(results)
        save_results(results, "/home/demisra/LatticeModel/lyapData_CSV/lyapData_$(M)p($p).csv")
    end
end
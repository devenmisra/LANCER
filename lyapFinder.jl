# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Julia 1.12.5
#     language: julia
#     name: julia-1.12
# ---

# %%
using LinearAlgebra
using Random
using CSV
using DataFrames
using Statistics
using Printf
using BenchmarkTools

BLAS.set_num_threads(1)

# %%
# ============================================================
# Transfer matrices
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

# %%
# ============================================================
# In-place block matrix actions
# ============================================================

function M1Action!(Z::Matrix{ComplexF64}, buf::Matrix{ComplexF64},
    B::Matrix{ComplexF64}, M::Int)
# Pull B elements into local registers
b11, b12 = B[1,1], B[1,2]
b21, b22 = B[2,1], B[2,2]

# FLIPPED LOOPS: k (column) is outer, i (row) is inner
@inbounds for k in axes(Z, 2)
for i in 1:M
r1 = 2i - 1
r2 = 2i
z1 = Z[r1, k]
z2 = Z[r2, k]
buf[r1, k] = b11*z1 + b12*z2
buf[r2, k] = b21*z1 + b22*z2
end
end
end

function M2Action!(Z::Matrix{ComplexF64}, buf::Matrix{ComplexF64},
    B::Matrix{ComplexF64}, M::Int)
n = 2 * M
b11, b12 = B[1,1], B[1,2]
b21, b22 = B[2,1], B[2,2]

@inbounds for k in axes(Z, 2)
# Main loop: rows 2 to n-1
for i in 1:M-1
r1 = 2i
r2 = 2i + 1
z1 = Z[r1, k]
z2 = Z[r2, k]
buf[r1, k] = b11*z1 + b12*z2
buf[r2, k] = b21*z1 + b22*z2
end
# Periodic boundary/end case
zn = Z[n, k]
z1 = Z[1, k]
buf[n, k] = b11*zn + b12*z1
buf[1, k] = b21*zn + b22*z1
end
end

# 1. Define a Struct to hold all pre-allocated memory for a single thread
struct LyapWorkspace
    U1::Vector{ComplexF64}
    U2::Vector{ComplexF64}
    row_theta::Vector{Float64}
    Z::Matrix{ComplexF64}
    buf1::Matrix{ComplexF64}
    buf2::Matrix{ComplexF64}
    γlist::Vector{Float64}

    function LyapWorkspace(M::Int)
        n = 2M
        new(
            zeros(ComplexF64, n), zeros(ComplexF64, n), 
            zeros(Float64, n), zeros(ComplexF64, n, n),
            zeros(ComplexF64, n, n), zeros(ComplexF64, n, n),
            zeros(Float64, n)
        )
    end
end

# 2. Optimized Action (Passes scalars to avoid Matrix allocations)
function TransfMAction_Inplace!(ws::LyapWorkspace, ex::Float64, M::Int)
    n = 2M
    # Pre-calculate scalars locally
    s1 = sqrt(1.0 + ex); t1 = 1.0 / sqrt(1.0 + 1.0 / ex)
    b1_11, b1_12 = s1, s1 * t1
    s2 = sqrt(1.0 + 1.0 / ex); t2 = 1.0 / sqrt(1.0 + ex)
    b2_11, b2_12 = s2, s2 * t2

    # All operations use the workspace fields (ws.buf1, ws.Z, etc.)
    @inbounds for k in 1:n, r in 1:n
        ws.buf1[r, k] = ws.U2[r] * ws.Z[r, k]
    end
    
    @inbounds for k in 1:n
        for i in 1:M-1
            r1, r2 = 2i, 2i + 1
            z1, z2 = ws.buf1[r1, k], ws.buf1[r2, k]
            ws.buf2[r1, k] = b2_11*z1 + b2_12*z2
            ws.buf2[r2, k] = b2_12*z1 + b2_11*z2
        end
        zn, z1 = ws.buf1[n, k], ws.buf1[1, k]
        ws.buf2[n, k] = b2_11*zn + b2_12*z1
        ws.buf2[1, k] = b2_12*zn + b2_11*z1
    end

    @inbounds for k in 1:n, r in 1:n
        ws.buf1[r, k] = ws.U1[r] * ws.buf2[r, k]
    end
    
    @inbounds for k in 1:n
        for i in 1:M
            r1, r2 = 2i-1, 2i
            z1, z2 = ws.buf1[r1, k], ws.buf1[r2, k]
            ws.Z[r1, k] = b1_11*z1 + b1_12*z2
            ws.Z[r2, k] = b1_12*z1 + b1_11*z2
        end
    end
end

# 3. Core calculation (No internal allocations)
function lyapfinder_single!(result_view, xlist, M, L, w, rng, ws::LyapWorkspace)
    n = 2M
    s = ceil(Int, L / w)
    Lcorr = w * s
    
    for c in eachindex(xlist)
        ex = exp(2.0 * xlist[c])
        rng_local = copy(rng) # Preserve disorder across x-values
        
        fill!(ws.Z, 0.0)
        for i in 1:n; ws.Z[i,i] = 1.0; end
        fill!(ws.γlist, 0.0)

        for i in 1:s
            for j in 1:w
                rand!(rng_local, ws.row_theta)
                for k in 1:n; ws.U1[k] = cis(2π * ws.row_theta[k]); end
                rand!(rng_local, ws.row_theta)
                for k in 1:n; ws.U2[k] = cis(2π * ws.row_theta[k]); end
                
                TransfMAction_Inplace!(ws, ex, M)
            end

            # In-Place Orthogonalization
            for k in 1:n
                col_norm = 0.0
                for row in 1:n; col_norm += abs2(ws.Z[row, k]); end
                col_norm = sqrt(col_norm)
                ws.γlist[k] += log(col_norm) / Lcorr
                
                for row in 1:n; ws.Z[row, k] /= col_norm; end
                
                for next_col in (k+1):n
                    dot_prod = ComplexF64(0.0)
                    for row in 1:n
                        dot_prod += conj(ws.Z[row, k]) * ws.Z[row, next_col]
                    end
                    for row in 1:n
                        ws.Z[row, next_col] -= dot_prod * ws.Z[row, k]
                    end
                end
            end
        end
        # Extract Localization Length
        neg_max = -Inf
        for val in ws.γlist
            if val < 0 && val > neg_max; neg_max = val; end
        end
        result_view[c] = M * (-neg_max)
    end
end

# 4. Multithreaded Wrapper
function lyapfinder_multi(xlist, M, L, w, n_real, master_seed)
    BLAS.set_num_threads(1)
    xN = length(xlist)
    γ_accum = Matrix{Float64}(undef, xN, n_real)
    
    master_rng = Xoshiro(master_seed)
    seeds = rand(master_rng, UInt64, n_real)

    # We use a pattern that creates the workspace INSIDE the thread loop
    # or uses a stable partition.
    
    # Let's use the 'Channel' or 'eachindex' approach to ensure safety:
    tasks_per_thread = ceil(Int, n_real / Threads.nthreads())
    
    Threads.@threads for tid in 1:Threads.nthreads()
        # Each thread creates its OWN workspace once
        ws = LyapWorkspace(M)
        
        # Calculate which realizations this specific thread will handle
        start_idx = (tid - 1) * tasks_per_thread + 1
        end_idx = min(tid * tasks_per_thread, n_real)
        
        for r in start_idx:end_idx
            rng = Xoshiro(seeds[r])
            # Pass our thread-local workspace 'ws' safely
            lyapfinder_single!(view(γ_accum, :, r), xlist, M, L, w, rng, ws)
        end
    end

    return [(xlist[c], mean(γ_accum[c, :]), std(γ_accum[c, :])) for c in 1:xN]
end
    
# ============================================================
# Helper: print results in a readable format
# ============================================================

function print_results(results::Vector{Tuple{Float64, Float64, Float64}})
println("\nx\t\tmean_γ\t\tstd_γ")
println("-"^50)
for (x, μ, σ) in results
@printf("%.4f\t\t%.6f\t%.6f\n", x, μ, σ)
end
end

# ============================================================
# Helper: save results to CSV
# ============================================================

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

# %%
MList = [20,40,60,80,100]

for M in MList
    results = lyapfinder_multi(
    collect(0.0:0.01:0.08),   # xlist
    M,                        # M
    1000000,                  # L
    20,                       # w
    384,                      # n_real
    42                        # seed
    )
    print_results(results)
    save_results(results, "/home/demisra/LatticeModel/lyapData$M.csv")
end
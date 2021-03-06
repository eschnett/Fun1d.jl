module Fun1d

using LinearAlgebra
using QuadGK
using Random

################################################################################

export lincom
"""
Calculate a linear combination
"""
function lincom(x1::S, y1::T, x2::S, y2::T, x::S) where {S,T}
    return T(x - x2) / T(x1 - x2) * y1 + T(x - x1) / T(x2 - x1) * y2
end

################################################################################

export Domain
"""
A domain (in the continuum)
"""
struct Domain{S}
    xmin::S
    xmax::S
end

function invariant(domain::Domain)
    return domain.xmin < domain.xmax
end

################################################################################

export Grid
"""
A discrete domain, with a particular resolution
"""
struct Grid{S}
    domain::Domain{S}
    ncells::Int
end

function invariant(grid::Grid)
    return invariant(grid.domain) && grid.ncells > 0
end

export location
"""
Grid point location
"""
function location(grid::Grid, n::Int)
    return lincom(1, grid.domain.xmin, grid.ncells + 1, grid.domain.xmax, n)
end

export spacing
"""
Grid spacing
"""
spacing(grid::Grid) = (grid.domain.xmax - grid.domain.xmin) / grid.ncells

export basis
"""
Evaluate the basis function for grid point `n` at point `x`
"""
function basis(grid::Grid{S}, n::Int, x::S) where {S}
    @assert 1 ≤ n ≤ grid.ncells + 1
    q = lincom(grid.domain.xmin, S(1), grid.domain.xmax, S(grid.ncells + 1), x)
    y1 = lincom(S(n - 1), S(0), S(n), S(1), q)
    y2 = lincom(S(n + 1), S(0), S(n), S(1), q)
    y = max(zero(S), min(y1, y2))
    return y
end

################################################################################

export GridFun
"""
A grid function, i.e. a discretized function living on a grid
"""
struct GridFun{S,T} <: AbstractArray{T,1}
    grid::Grid{S}
    values::Vector{T}

    function GridFun{S,T}(grid::Grid{S}, values::Vector{T}) where {S,T}
        @assert length(values) == grid.ncells + 1
        return new{S,T}(grid, values)
    end
end

function invariant(f::GridFun)
    return invariant(f.grid) && length(f.values) == f.grid.ncells + 1
end

Base.:(==)(f::GridFun, g::GridFun) = f.grid == g.grid && f.values == g.values

function GridFun(grid::Grid{S}, values::Vector{T}) where {S,T}
    return GridFun{S,T}(grid, values)
end

# Iteration
Base.eltype(::Type{GridFun{S,T}}) where {S,T} = T
Base.isempty(::GridFun) = false
Base.iterate(f::GridFun, state...) = iterate(f.values, state...)
# Base.length(f::GridFun) = length(f.values)

# Indexing
Base.firstindex(f::GridFun) = firstindex(f.values)
Base.getindex(f::GridFun, i) = getindex(f.values, i)
Base.lastindex(f::GridFun) = lastindex(f.values)
Base.setindex!(f::GridFun, v, i) = setindex!(f.values, v, i)

# Abstract Array
Base.size(f::GridFun) = size(f.values)
Base.IndexStyle(::GridFun) = IndexLinear()
Base.similar(f::GridFun) = GridFun(f.grid, similar(f.values))
function Base.similar(f::GridFun, ::Type{T}) where {T}
    return GridFun(f.grid, similar(f.values, T))
end
Base.similar(f::GridFun, ::Dims) = similar(f)
Base.similar(f::GridFun, ::Dims, ::Type{T}) where {T} = similar(f, T)

# Broadcasting
Base.BroadcastStyle(::Type{<:GridFun}) = Broadcast.ArrayStyle{GridFun}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GridFun}},
                      ::Type{T}) where {T}
    f = find_gridfun(bc)
    return similar(f, T)
end
find_gridfun(bc::Base.Broadcast.Broadcasted) = find_gridfun(bc.args)
find_gridfun(args::Tuple) = find_gridfun(find_gridfun(args[1]), Base.tail(args))
find_gridfun(x) = x
find_gridfun(::Tuple{}) = nothing
find_gridfun(a::GridFun, rest) = a
find_gridfun(::Any, rest) = find_gridfun(rest)

# Others
function Base.map(fun, f::GridFun, gs::GridFun...)
    @assert all(g.grid == f.grid for g in gs)
    return GridFun(f.grid, map(fun, f.values, (g.values for g in gs)...))
end

function Base.rand(rng::AbstractRNG, ::Type{GridFun{S,T}},
                   grid::Grid{S}) where {S,T}
    return GridFun(grid, rand(rng, T, grid.ncells + 1))
end
function Base.rand(::Type{GridFun{S,T}}, grid::Grid{S}) where {S,T}
    return rand(Random.GLOBAL_RNG, GridFun{S,T}, grid)
end

function Base.zero(::Type{GridFun{S,T}}, grid::Grid{S}) where {S,T}
    return GridFun(grid, zeros(T, grid.ncells + 1))
end
Base.zero(f::GridFun{S,T}) where {S,T} = zero(GridFun{S,T}, f.grid)

Base.iszero(f::GridFun) = all(iszero, f.values)

Base.:+(f::GridFun) = GridFun(f.grid, +f.values)
Base.:-(f::GridFun) = GridFun(f.grid, -f.values)

function Base.:+(f::GridFun, g::GridFun)
    @assert f.grid == g.grid
    return GridFun(f.grid, f.values + g.values)
end
function Base.:-(f::GridFun, g::GridFun)
    @assert f.grid == g.grid
    return GridFun(f.grid, f.values - g.values)
end

Base.:*(f::GridFun, a::Number) = GridFun(f.grid, f.values * a)
Base.:*(a::Number, f::GridFun) = GridFun(f.grid, a * f.values)
Base.:/(f::GridFun, a::Number) = GridFun(f.grid, f.values / a)
Base.:\(a::Number, f::GridFun) = GridFun(f.grid, a \ f.values)

################################################################################

export sample
"""
Sample a function at grid points
"""
function sample(::Type{T}, grid::Grid{S}, fun) where {S,T}
    values = T[fun(location(grid, n)) for n in 1:(grid.ncells + 1)]
    return GridFun(grid, values)
end

export project
"""
Project a function onto the basis functions
"""
function project(::Type{T}, grid::Grid{S}, fun) where {S,T}
    atol = eps(T)^(T(3) / 4)
    n = grid.ncells + 1
    dx = spacing(grid)

    fbvalues = Array{T}(undef, n)
    for i in 1:n
        x0 = location(grid, i)
        xmin = max(grid.domain.xmin, x0 - dx)
        xmax = min(grid.domain.xmax, x0 + dx)
        I, E = quadgk(x -> fun(x) * basis(grid, i, x), xmin, xmax; atol=atol)
        fbvalues[i] = I
    end

    bbdiag = Array{S}(undef, n)
    bbdiag[1] = dx / 3
    bbdiag[2:(end - 1)] .= 2 * dx / 3
    bbdiag[end] = dx / 3
    bbsdiag = Array{S}(undef, n - 1)
    bbsdiag[1:end] .= dx / 6
    bbmatrix = SymTridiagonal(bbdiag, bbsdiag)

    fvalues = bbmatrix \ fbvalues
    return GridFun(grid, fvalues)
end

export evaluate
"""
Evaluate a grid function at an arbitrary point
"""
function evaluate(f::GridFun{S,T}, x::S) where {S,T}
    q = lincom(f.grid.domain.xmin, S(1), f.grid.domain.xmax,
               S(f.grid.ncells + 1), x)
    @assert q ≥ 1 && q ≤ f.grid.ncells + 1
    n = min(f.grid.ncells, floor(Int, q))
    y1 = f.values[n] * basis(f.grid, n, x)
    y2 = f.values[n + 1] * basis(f.grid, n + 1, x)
    y = y1 + y2
    return y::T
end

# TODO: norm2 (both for continuum and grid functions)

################################################################################

export integrate
"""
Integrate a grid function over the whole domain
"""
function integrate(f::GridFun{S,T}) where {S,T}
    dx = spacing(f.grid)
    n = f.grid.ncells + 1
    int = zero(T)
    int += f.values[1] / 2
    for i in 2:(n - 1)
        int += f.values[i]
    end
    int += f.values[n] / 2
    int *= dx
    return int
end

export norm2
"""
L2-norm of a grid function
"""
norm2(f::GridFun) = sqrt(integrate(map(abs2, f)))

#function LinearAlgebra.norm(f::GridFun{S,T})where{S,T}

export deriv
"""
Derivative of a grid function
"""
function deriv(f::GridFun{S,T}) where {S,T}
    dx = spacing(f.grid)
    n = f.grid.ncells + 1
    dvalues = Array{T}(undef, n)
    dvalues[1] = (f.values[2] - f.values[1]) / dx
    for i in 2:(n - 1)
        dvalues[i] = (f.values[i + 1] - f.values[i - 1]) / 2dx
    end
    dvalues[n] = (f.values[n] - f.values[n - 1]) / dx
    return GridFun(f.grid, dvalues)
end

export deriv2
"""
Second derivative of a grid function
"""
function deriv2(f::GridFun{S,T}) where {S,T}
    dx = spacing(f.grid)
    n = f.grid.ncells + 1
    d2values = Array{T}(undef, n)
    for i in 2:(n - 1)
        d2values[i] = (f.values[i - 1] - 2 * f.values[i] + f.values[i + 1]) /
                      dx^2
    end
    d2values[1] = d2values[2]
    d2values[n] = d2values[n - 1]
    return GridFun(f.grid, d2values)
end

export deriv2_sphere
"""
Second derivative of a grid function in spherical coordinates,
including the metric term (∂ᵣᵣ + 2/r ∂ᵣ)
"""
function deriv2_sphere(f::GridFun{S,T}) where {S,T}
    dx = spacing(f.grid)
    n = f.grid.ncells + 1
    d2values = Array{T}(undef, n)
    # Use L'Hôpital's rule at r = 0: limit (r→0) f′(r) / r = f′′(0)
    d2values[1] = 6 * (f.values[2] - f.values[1]) / dx^2
    for i in 2:(n - 1)
        r = dx * (i - 1)
        d2values[i] = (f.values[i - 1] - 2 * f.values[i] + f.values[i + 1]) /
                      dx^2 + 2 / r * (f.values[i + 1] - f.values[i - 1]) / 2dx
    end
    r = dx * (n - 1)
    d2values[n] = (f.values[n - 2] - 2 * f.values[n - 1] + f.values[n]) / dx^2 +
                  2 / r * (f.values[n] - f.values[n - 1]) / dx
    return GridFun(f.grid, d2values)
end

function deriv2_sphere_unused(f::GridFun{S,T}) where {S,T}
    dx = spacing(f.grid)
    n = f.grid.ncells + 1
    # 1/r² ∂_r r² ∂ᵣ
    # 1. cell-centred (forward) derivative:
    #    u[i] := (u[i+1] - u[i]) / dx
    # 2. multiply by metric term:
    #    u[i] := ((i+1/2) dx)^2 * u[i]
    # 3. vertex-centred (backward) derivative:
    #    u[i] := (u[i] - u[i-1]) / dx
    # 4. divide by metric term:
    #    u[i] := 1/(i dx)^2 * u[i]
    d2values = Array{T}(undef, n)
    d2values[1] = 6 * (f.values[2] - f.values[1]) / dx^2
    for i in 2:(n - 1)
        j = i - 1
        d2values[i] = ((1 + (1 - 4 * j) / T(4 * j^2)) * f.values[i - 1] -
                       (2 + 2 / T(4 * j^2)) * f.values[i] +
                       (1 + (1 + 4 * j) / T(4 * j^2)) * f.values[i + 1]) / dx^2
    end
    j = n - 1
    d2values[n] = ((1 + (1 - 2 * j) / T(2 * j^2)) * f.values[n - 2] -
                   (2 + 2 / T(2 * j^2)) * f.values[n - 1] +
                   (1 + (1 + 2 * j) / T(2 * j^2)) * f.values[n]) / dx^2
    return GridFun(f.grid, d2values)
end

################################################################################

end

module Fun1d

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
struct GridFun{S,T}
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

export evaluate
"""
Evaluate a grid function at an arbitrary point
"""
function evaluate(f::GridFun{S}, x::S) where {S}
    q = lincom(f.grid.domain.xmin, S(1), f.grid.domain.xmax,
               S(f.grid.ncells + 1), x)
    @assert q ≥ 1 && q ≤ f.grid.ncells + 1
    n = min(f.grid.ncells, floor(Int, q))
    y1 = f.values[n] * basis(f.grid, n, x)
    y2 = f.values[n + 1] * basis(f.grid, n + 1, x)
    y = y1 + y2
    return y
end

################################################################################

end

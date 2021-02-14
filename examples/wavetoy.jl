module WaveToy

using DifferentialEquations
using Fun1d
# using Random

struct Wave{S,T} <: AbstractArray{T,2}
    ϕ::GridFun{S,T}
    ψ::GridFun{S,T}
end

# Iteration
Base.IteratorSize(::Type{<:Wave}) = Iterators.HasShape{2}()
Base.eltype(::Type{Wave{S,T}}) where {S,T} = T
Base.isempty(x::Wave) = isempty(x.ϕ)
function Base.iterate(x::Wave, state...)
    return iterate(Iterators.flatten((x.ϕ, x.ψ)), state...)
end
# Base.length(x::Wave) = 2 * length(x.ϕ)
Base.size(x::Wave) = (length(x.ϕ), 2)
Base.size(x::Wave, d) = size(x)[d]

# Indexing
function lin2cart(x::Wave, i::Number)
    n = length(x.ϕ)
    return (i - 1) % n + 1, (i - 1) ÷ n + 1
end
Base.firstindex(x::Wave) = error("not implemented")
Base.getindex(x::Wave, i) = getindex(x, i.I...)
Base.getindex(x::Wave, i::Number) = getindex(x, lin2cart(x, i)...)
Base.getindex(x::Wave, i, j) = getindex((x.ϕ, x.ψ)[j], i)
Base.lastindex(x::Wave) = error("not implemented")
Base.setindex!(x::Wave, v, i) = setindex!(x, v, i.I...)
Base.setindex!(x::Wave, v, i::Number) = setindex!(x, v, lin2cart(x, i))
Base.setindex!(x::Wave, v, i, j) = setindex!((x.ϕ, x.ψ)[j], v, i)

# Abstract Array
Base.IndexStyle(::Wave) = IndexCartesian()
Base.similar(x::Wave) = Wave(similar(x.ϕ), similar(x.ψ))
function Base.similar(x::Wave, ::Type{T}) where {T}
    return Wave(similar(x.ϕ, T), similar(x.ψ, T))
end
Base.similar(x::Wave, ::Dims) = similar(x)
Base.similar(x::Wave, ::Dims, ::Type{T}) where {T} = similar(x, T)

# Broadcasting
Base.BroadcastStyle(::Type{<:Wave}) = Broadcast.ArrayStyle{Wave}()
function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{Wave}},
                      ::Type{T}) where {T}
    x = find_wave(bc)
    return similar(x, T)
end
find_wave(bc::Base.Broadcast.Broadcasted) = find_wave(bc.args)
find_wave(args::Tuple) = find_wave(find_wave(args[1]), Base.tail(args))
find_wave(x) = x
find_wave(::Tuple{}) = nothing
find_wave(a::Wave, rest) = a
find_wave(::Any, rest) = find_wave(rest)

# Others
function Base.map(fun, x::Wave, ys::Wave...)
    return Wave(map(fun, x.ϕ, (y.ϕ for y in ys)...),
                map(fun, x.ψ, (y.ψ for y in ys)...))
end

# function Base.rand(rng::AbstractRNG, ::Random.SamplerType{Wave{T}}) where {T}
#     return Wave{T}(rand(rng, T), rand(rng, T))
# end

Base.zero(::Type{<:Wave}) = error("not implemented")
Base.zero(x::Wave) = Wave(zero(x.ϕ), zero(x.ψ))

Base.:+(x::Wave) = map(+, x)
Base.:-(x::Wave) = map(-, x)

Base.:+(x::Wave, y::Wave) = map(+, x, y)
Base.:-(x::Wave, y::Wave) = map(-, x, y)

Base.:*(x::Wave, a::Number) = map(b -> b * a, x)
Base.:*(a::Number, x::Wave) = map(b -> a * b, x)
Base.:/(x::Wave, a::Number) = map(b -> b / a, x)
Base.:\(a::Number, x::Wave) = map(b -> a \ b, x)

function setup(::Type{S}) where {S}
    domain = Domain{S}(0, 1)
    grid = Grid(domain, 20)
    return grid
end

function init(::Type{T}, grid::Grid) where {T}
    println("WaveToy.init")
    # u(t,x) = sin(π t) cos(π x)
    ϕ = project(T, grid, x -> 0)
    ψ = project(T, grid, x -> π * cospi(x))
    return Wave(ϕ, ψ)
end

function rhs(state::Wave, param, t)
    println("WaveToy.rhs t=$t")
    ϕdot = state.ψ
    ψdot = deriv2(state.ϕ)
    staterhs = Wave(ϕdot, ψdot)
    return staterhs::Wave
end

function test()
    T = Float64
    grid = setup(T)
    state0 = init(T, grid)::Wave
    @show state0
    @show 2 * state0
    @show state0 + state0
    nothing
end

function main()
    T = Float64
    grid = setup(T)
    state = init(T, grid)::Wave
    tspan = T[0, 1]
    atol = eps(T)^(T(3) / 4)
    prob = ODEProblem(rhs, state, tspan)
    sol = solve(prob, Tsit5(); abstol=atol)
    return sol
end

# include("examples/wavetoy.jl")
# sol = WaveToy.main();
# using Plots
# plot([sol(t) for t in LinRange(0, 1, 11)])

end

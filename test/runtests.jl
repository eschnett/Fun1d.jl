using Fun1d
using QuadGK
using Random
using Test

# Random rationals
function Base.rand(rng::AbstractRNG,
                   ::Random.SamplerType{Rational{T}}) where {T}
    return Rational{T}(T(rand(rng, -1000:1000)) // 1000)
end

const Rat128 = Rational{Int128}

@testset "Domain" begin
    S = Rat128
    domain = Domain{S}(0, 1)
    @test Fun1d.invariant(domain)
end

@testset "Grid" begin
    S = Rat128
    domain = Domain{S}(0, 1)
    grid = Grid(domain, 10)
    @test Fun1d.invariant(grid)

    # Test some specific basis functions
    dx = spacing(grid)
    @test dx > 0

    @test basis(grid, 1, 0 * dx) == 1
    @test basis(grid, 1, 1 // 2 * dx) == 1 // 2
    @test basis(grid, 1, dx) == 0

    @test basis(grid, 2, 0 * dx) == 0
    @test basis(grid, 2, 1 // 2 * dx) == 1 // 2
    @test basis(grid, 2, dx) == 1
    @test basis(grid, 2, 3 // 2 * dx) == 1 // 2
    @test basis(grid, 2, 2 * dx) == 0

    # Test that basis functions form a position basis: At grid point
    # `i`, the `i`-th basis function must be `1`, all other basis
    # function must be `0`.
    for i in 1:(grid.ncells + 1)
        x = location(grid, i)
        for j in 1:(grid.ncells + 1)
            y = basis(grid, j, x)
            @test y == (i == j)
        end
    end

    # Test that basis functions form a partition of unity: At each
    # point in the domain, the basis functions must sum up to `1`
    for i in 1:10
        x = lincom(1, grid.domain.xmin, 1000000, grid.domain.xmax,
                   rand(1:1000000))
        sumy = zero(S)
        for j in 1:(grid.ncells + 1)
            y = basis(grid, j, x)
            @test 0 ≤ y ≤ 1
            sumy += y
        end
        @test sumy == 1
    end

    # TODO: test compact support
end

Random.seed!(0)
@testset "GridFun" begin
    S = Rat128
    T = Rat128
    domain = Domain{S}(0, 1)
    grid = Grid(domain, 10)
    for n in 1:10
        f = rand(GridFun{S,T}, grid)
        g = rand(GridFun{S,T}, grid)
        h = rand(GridFun{S,T}, grid)
        z = zero(GridFun{S,T}, grid)
        a = rand(T)
        b = rand(T)

        @test Fun1d.invariant(f)
        @test Fun1d.invariant(g)
        @test Fun1d.invariant(h)
        @test Fun1d.invariant(z)

        @test iszero(z)
        @test f == f
        @test +f == f
        @test f + z == f
        @test z + f == f
        @test -(-f) == f
        @test f + (-f) == z
        @test f - g == f + (-g)
        @test f + g == g + f
        @test (f + g) + h == f + (g + h)

        @test zero(T) * f == z
        @test one(T) * f == f
        @test (-one(T)) * f == -f
        @test a * f == f * a
        @test a * (f + g) == a * f + a * g
        @test (a + b) * f == a * f + b * f
        @test (a * b) * f == a * (b * f)
        if !iszero(a)
            @test f / a == f * inv(a)
            @test a \ f == inv(a) * f
        end
    end
end

Random.seed!(0)
@testset "GridFun sample/evaluate" begin
    S = Rat128
    T = Rat128
    domain = Domain{S}(0, 1)
    grid = Grid(domain, 10)

    m = rand(T)
    b = rand(T)
    fun(x) = m * x + b
    gf = sample(T, grid, fun)
    @test Fun1d.invariant(gf)

    for n in 1:10
        x = lincom(1, grid.domain.xmin, 1000000, grid.domain.xmax,
                   rand(1:1000000))
        @test evaluate(gf, x)::T == m * x + b
    end
end

Random.seed!(0)
@testset "GridFun convergence" begin
    S = Float64
    T = Float64
    domain = Domain{S}(0, 1)
    grid1 = Grid(domain, 20)
    grid2 = Grid(domain, 40)

    fun(x) = sinpi(x)
    gf1 = sample(T, grid1, fun)
    gf2 = sample(T, grid2, fun)

    atol = eps(T)^(T(3) / 4)
    function int(f)
        I, E = quadgk(f, domain.xmin, domain.xmax; atol=atol)
        return I
    end
    function norm2(f)
        return sqrt(int(x -> abs2(f(x))) / (domain.xmax - domain.xmin))
    end

    e1 = norm2(x -> evaluate(gf1, x) - fun(x))
    e2 = norm2(x -> evaluate(gf2, x) - fun(x))
    ratio = e1 / e2
    @test isapprox(ratio, 4; rtol=0.01)
end

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

Random.seed!(0)
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
@testset "GridFuns are vectors" begin
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

        @test map(identity, f) == f
        @test map(x -> x + b, map(x -> a * x, f)) == map(x -> a * x + b, f)
        @test map(+, f, g) == f + g

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
        @test evaluate(gf, x) == fun(x)
    end
end

Random.seed!(0)
@testset "GridFun project/evaluate" begin
    S = Float64
    T = Float64
    domain = Domain{S}(0, 1)
    grid = Grid(domain, 10)

    m = rand(T)
    b = rand(T)
    fun(x) = m * x + b
    gf = project(T, grid, fun)
    @test Fun1d.invariant(gf)

    atol = eps(T)^(T(3) / 4)
    for n in 1:10
        x = lincom(1, grid.domain.xmin, 1000000, grid.domain.xmax,
                   rand(1:1000000))
        @test isapprox(evaluate(gf, x), fun(x); atol=atol)
    end
end

Random.seed!(0)
@testset "GridFun convergence" begin
    S = Float64
    T = Float64
    domain = Domain{S}(0, 1)
    grid1 = Grid(domain, 20)
    grid2 = Grid(domain, 40)

    atol = eps(T)^(T(3) / 4)
    function int(f)
        I, E = quadgk(f, domain.xmin, domain.xmax; atol=atol)
        return I
    end
    norm2(f) = sqrt(int(x -> abs2(f(x))))

    fun(x) = sinpi(x)
    gf1 = sample(T, grid1, fun)
    gf2 = sample(T, grid2, fun)

    e1 = norm2(x -> evaluate(gf1, x) - fun(x))
    e2 = norm2(x -> evaluate(gf2, x) - fun(x))
    ratio = e1 / e2
    @test isapprox(ratio, 4; rtol=0.01)
end

@testset "GridFun derivative linearity" begin
    S = Rat128
    T = Rat128
    domain = Domain{S}(0, 1)
    grid = Grid(domain, 10)

    for n in 1:10
        f = rand(GridFun{S,T}, grid)
        g = rand(GridFun{S,T}, grid)
        z = zero(GridFun{S,T}, grid)
        a = rand(T)

        @test deriv(z) == z
        @test deriv(f + g) == deriv(f) + deriv(g)
        @test deriv(a * f) == a * deriv(f)

        @test deriv2(z) == z
        @test deriv2(f + g) == deriv2(f) + deriv2(g)
        @test deriv2(a * f) == a * deriv2(f)

        @test deriv2_sphere(z) == z
        @test deriv2_sphere(f + g) == deriv2_sphere(f) + deriv2_sphere(g)
        @test deriv2_sphere(a * f) == a * deriv2_sphere(f)
    end
end

@testset "GridFun integral accuracy" begin
    S = Float64
    T = Float64
    domain = Domain{S}(0, 1)
    grid1 = Grid(domain, 20)
    grid2 = Grid(domain, 40)

    atol = eps(T)^(T(3) / 4)

    fun(x) = sinpi(x)
    gf1 = sample(T, grid1, fun)
    gf2 = sample(T, grid2, fun)

    i0 = 2 / T(π)
    i1 = integrate(gf1)
    i2 = integrate(gf2)
    iratio = (i1 - i0) / (i2 - i0)
    @test isapprox(iratio, 4; atol=0.01)

    n0 = 1 / sqrt(T(2))
    n1 = norm2(gf1)
    n2 = norm2(gf2)
    # nratio = (n1 - n0) / (n2 - n0)
    # @test isapprox(nratio, 4; atol=0.01)
    # Strangely, this result is much more accurate than expected. I
    # won't complain.
    @test isapprox(n1, n0; atol=atol)
    @test isapprox(n2, n0; atol=atol)
end

@testset "GridFun derivative accuracy" begin
    S = Float64
    T = Float64
    domain = Domain{S}(0, 1)
    grid1 = Grid(domain, 20)
    grid2 = Grid(domain, 40)

    atol = eps(T)^(T(3) / 4)
    dx = spacing(grid1)
    function int(f)
        # Omit outer boundary to get good convergence
        I, E = quadgk(f, domain.xmin + dx, domain.xmax - dx; atol=atol)
        return I
    end
    norm2(f) = sqrt(int(x -> abs2(f(x))))

    fun(x) = sinpi(x)
    gf1 = sample(T, grid1, fun)
    gf2 = sample(T, grid2, fun)

    dfun(x) = π * cospi(x)
    dgf1 = deriv(gf1)
    dgf2 = deriv(gf2)

    d2fun(x) = -T(π)^2 * sinpi(x)
    d2gf1 = deriv2(gf1)
    d2gf2 = deriv2(gf2)

    # ef1 = norm2(x -> evaluate(gf1, x) - fun(x))
    # ef2 = norm2(x -> evaluate(gf2, x) - fun(x))
    edf1 = norm2(x -> evaluate(dgf1, x) - dfun(x))
    edf2 = norm2(x -> evaluate(dgf2, x) - dfun(x))
    ed2f1 = norm2(x -> evaluate(d2gf1, x) - d2fun(x))
    ed2f2 = norm2(x -> evaluate(d2gf2, x) - d2fun(x))

    ratio = edf1 / edf2
    @test isapprox(ratio, 4; rtol=0.01)
    ratio2 = ed2f1 / ed2f2
    @test isapprox(ratio2, 4; rtol=0.01)
end

@testset "GridFun spherical derivative accuracy" begin
    S = Float64
    T = Float64
    domain = Domain{S}(0, 1)
    grid1 = Grid(domain, 20)
    grid2 = Grid(domain, 40)

    atol = eps(T)^(T(3) / 4)
    dx = spacing(grid1)
    function int(f)
        # Omit outer boundary to get good convergence
        I, E = quadgk(f, domain.xmin, domain.xmax - dx; atol=atol)
        return I
    end
    # norm2(f) = sqrt(int(r -> 4 * T(π) * r^2 * abs2(f(r))))
    norm2(f) = sqrt(int(r -> abs2(f(r))))

    fun(r) = cospi(r)
    gf1 = sample(T, grid1, fun)
    gf2 = sample(T, grid2, fun)

    d2fun(r) = -T(π)^2 * (cospi(r) + 2 * sinc(r))
    d2gf1 = deriv2_sphere(gf1)
    d2gf2 = deriv2_sphere(gf2)

    ed2f1 = norm2(r -> evaluate(d2gf1, r) - d2fun(r))
    ed2f2 = norm2(r -> evaluate(d2gf2, r) - d2fun(r))

    ratio2 = ed2f1 / ed2f2
    @test isapprox(ratio2, 4; rtol=0.02)
end

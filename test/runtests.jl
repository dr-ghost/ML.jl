using ML, Test

@testset "ML.jl" begin
    @test ML.greet() == "Hello ML Warriors"
end
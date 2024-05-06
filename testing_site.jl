using LinearAlgebra

include("core.jl")
#include("dataloader.jl")

# a = ones((3))
# # a[1, 1, 1] = 5
# b = diagm(a)
# println(b)

function relu(x, y)
    return max.(x, y)
end


function backward(x, y)
    Jx = diagm(isless.(y, x))
    Jy = diagm(isless.(x, y))
    return tuple(Jx', Jy')
end

a = [1 2 3; 1 2 3;;;1 2 3; 1 2 3;;;]
b = 2

#println(isless.(a, b))
#println(isless.(b, a))

if b > 1
    println(1)
else
    println(2)
end
# tuple(g * n * x ^ (n-1), g * log(abs(x)) * x ^ n)

# \Added

#=
using LinearAlgebra
Wh  = Variable(randn(10,2), name="wh")
Wo  = Variable(randn(2,10), name="wo")
# Wo  = Variable(randn(2,10), name="wo")
x = []
y = []
dataset = 10
for _ in range(1, step=1, length=dataset)
    x1 = rand(Float64)
    x2 = rand(Float64)
    y1 = 5*x1 + 6*x2
    y2 = 3*x1 + 15*x2
    push!(x, Variable([x1, x2], name="x"))
    push!(y, Variable([y1, y1], name="y"))
end
#y = Variable([0.064, 0.33], name="y")
#y = Variable([1, 2], name="y")
losses = Float64[]
=#

#=
function net(x, wh, wo, y)
    x̂ = dense(wh, x)
    x̂.name = "x̂"
    ŷ = dense(wo, x̂)
    ŷ.name = "ŷ"
    E = mean_squared_loss(y, ŷ)
    E.name = "loss"

    return topological_sort(E)
end

for _ in range(1, step=1, length=100)
    loss = 0
    for i in range(1, step=1, length=dataset)
        graph = net(x[i], Wh, Wo, y[i])

        currentloss = forward!(graph)
        backward!(graph)
        Wh.output -= 0.01Wh.gradient
        Wo.output -= 0.01Wo.gradient
        loss += currentloss[1]
        push!(losses, first(currentloss))
    end
    println("Current loss: ", loss/dataset)
end

for i in range(1, step=1, length=dataset)
    x̂ = dense(Wh, x[i])
    x̂.name = "x̂"
    ŷ = dense(Wo, x̂)
    ŷ.name = "ŷ"
    answ = forward!(topological_sort(ŷ))
    # println(i)
    println("y = ", y[i].output, " answ = ", answ)
end
=#

#train_data = load_train_data()
#test_data = load_test_data()

using LinearAlgebra
Wh  = Variable(randn(10,2), name="wh")
Wo  = Variable(randn(2,10), name="wo")


x = []
y = []
dataset = 10
for _ in range(1, step=1, length=dataset)
    x1 = rand(Float64)
    x2 = rand(Float64)
    y1 = 5*x1 + 6*x2
    y2 = 3*x1 + 15*x2
    push!(x, Variable([x1, x2], name="x"))
    push!(y, Variable([y1, y1], name="y"))
end
#y = Variable([0.064, 0.33], name="y")
#y = Variable([1, 2], name="y")

losses = Float64[]

function net(x, wh, wo, y)
    x̂ = dense(wh, x)
    x̂.name = "x̂"
    ŷ = dense(wo, x̂)
    ŷ.name = "ŷ"
    E = mean_squared_loss(y, ŷ)
    E.name = "loss"

    return topological_sort(E)
end

"""
for _ in range(1, step=1, length=100)
    loss = 0
    for i in range(1, step=1, length=dataset)
        graph = net(x[i], Wh, Wo, y[i])

        currentloss = forward!(graph)
        backward!(graph)
        Wh.output -= 0.01Wh.gradient
        Wo.output -= 0.01Wo.gradient
        loss += currentloss[1]
        push!(losses, first(currentloss))
    end
    println("Current loss: ", loss/dataset)
end

for i in range(1, step=1, length=dataset)
    x̂ = dense(Wh, x[i])
    x̂.name = "x̂"
    ŷ = dense(Wo, x̂)
    ŷ.name = "ŷ"
    answ = forward!(topological_sort(ŷ))
    println("y = ", y[i].output, " answ = ", answ)
end"""

test = [1, 56, 3, 4]
test1 = [1, 56, 3, 4]
println(findmax(test)[2] == findmax(test1)[2])

#=function pooling(x::Union{Array, Matrix{Float64}}, kernel::Int=2, stride::Int=kernel)
    channels = size(x,3)
    output_shape = (size(x,1) - kernel) / stride + 1
    output_shape = trunc(Int, output_shape)
    output = zeros((output_shape, output_shape, channels))
    for channel in 1:channels
        for i in 1:stride:size(x,1)
            for j in 1:stride:size(x,2)
                output[fld(i,stride)+1, fld(j,stride)+1, channel] = maximum(x[i:i+kernel-1,j:j+kernel-1,channel])
            end
        end
    end      
    return output
end=#
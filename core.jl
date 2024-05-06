import Base: +, -, *, /, ^, sin, sum, abs, max, log, size
import Base: show, summary
import LinearAlgebra: mul!

abstract type GraphNode end

abstract type Operator <: GraphNode end

struct Constant <: GraphNode
    output :: Union{Int64, Float64, Matrix{Float64}}
end

mutable struct Variable <: GraphNode
    output :: Union{Int64, Float64, Matrix{Float64}, Array{Float64}}
    gradient :: Union{Float64, Matrix{Float64}, Array{Float64}, Nothing, BitArray}
    Variable(output) = new(output, nothing)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Tuple#{Union{Variable, Constant, ScalarOperator}...}
    output :: Union{Int64, Float64, Matrix{Float64}, Array{Float64}, Nothing}
    gradient :: Union{Float64, Matrix{Float64}, Array{Float64}, Nothing, BitArray}
    ScalarOperator(fun, inputs...) = new{typeof(fun)}(inputs, nothing, nothing)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Tuple#{Union{Variable, Constant, ScalarOperator, BroadcastedOperator}...}
    output :: Union{Int64, Float64, Matrix{Float64}, Array{Float64}, Nothing}
    gradient :: Union{Float64, Matrix{Float64}, Array{Float64}, Nothing, BitArray}
    BroadcastedOperator(fun, inputs...) = new{typeof(fun)}(inputs, nothing, nothing)
end

#import Base: show, summary
#show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");
#show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");
#show(io::IO, x::Constant) = print(io, "const ", x.output)
#show(io::IO, x::Variable) = begin
#    print(io, "var ", x.name);
#    print(io, "\n ┣━ ^ "); summary(io, x.output)
#    print(io, "\n ┗━ ∇ ");  summary(io, x.gradient)
#end

function visit(node::GraphNode, visited::Set, order::Vector)
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end
    
function visit(node::Operator, visited::Set, order::Vector)
    if node ∈ visited
    else
        push!(visited, node)
        @inbounds for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode)
    visited = Set()
    order = Vector()
    visit(head, visited, order)
    return order
end

reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) =
    node.output = forward(node, [input.output for input in node.inputs]...)

function forward!(order::Vector)
    @inbounds for node in order
        compute!(node)
        reset!(node)
    end
    
    return last(order).output
end

update!(node::Constant, gradient::Union{Float64, Matrix{Float64}, Array{Float64}, Nothing, BitArray}) = nothing
update!(node::GraphNode, gradient::Union{Float64, Matrix{Float64}, Array{Float64}, Nothing, BitArray}) = if isnothing(node.gradient)
    node.gradient = gradient else node.gradient .+= gradient
end

function backward!(order::Vector; seed::Float64=1.0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    @inbounds for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end

#import Base: ^
^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x ^ (n-1), g * log(abs(x)) * x ^ n)

#import Base: sin
sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = return sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = tuple(g * cos(x))

#import Base: *
#import LinearAlgebra: mul!
# x * y (aka matrix multiplication)
*(A::GraphNode, x::GraphNode) = BroadcastedOperator(mul!, A, x)
forward(::BroadcastedOperator{typeof(mul!)}, A, x) = return A * x
backward(::BroadcastedOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

# x .* y (element-wise multiplication)
Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = BroadcastedOperator(*, x, y)
forward(::BroadcastedOperator{typeof(*)}, x, y) = return x .* y
backward(node::BroadcastedOperator{typeof(*)}, x, y, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(y .* 𝟏)
    Jy = diagm(x .* 𝟏)
    #println("*", size(Jx' * g))
    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = let
    #println("-", size(g))
    tuple(g,-g)
end

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = let
    #println("+", size(g))
    tuple(g, g)
end

#import Base: sum
sum(x::GraphNode) = BroadcastedOperator(sum, x::GraphNode)
forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = let
    𝟏 = ones(length(x))
    J = 𝟏'
    #println("sum", size(J' * g))
    tuple(J' * g)
end

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = BroadcastedOperator(/, x, y)
forward(::BroadcastedOperator{typeof(/)}, x, y) = return x ./ y
backward(node::BroadcastedOperator{typeof(/)}, x, y::Real, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(𝟏 ./ y)
    Jy = (-x ./ y .^2)
    tuple(Jx' * g, Jy' * g)
end

#import Base: max
Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = BroadcastedOperator(max, x, y)
forward(::BroadcastedOperator{typeof(max)}, x, y) = return max.(x, y)
backward(::BroadcastedOperator{typeof(max)}, x, y, g) = let
    if size(x, 3) == 1
        Jx = diagm(isless.(y, x))
        Jy = diagm(isless.(x, y))
    else
        Jx = isless.(y, x)
        Jy = isless.(x, y)
    end
    #println(size(Jx), " ", size(g))
    tuple(Jx' * g, Jy' * g)
end

# Added

Base.Broadcast.broadcasted(^, x::GraphNode, n::GraphNode) = BroadcastedOperator(^, x, n)
forward(::BroadcastedOperator{typeof(^)}, x, n) = return x.^n
backward(node::BroadcastedOperator{typeof(^)}, x, n, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(n * x .^ (n-1) .* 𝟏)
    Jy = diagm(log.(abs.(x)) .* x .^ n .* 𝟏)
    tuple(Jx' * g, Jy' * g)
end

#import Base: abs
Base.Broadcast.broadcasted(abs, x::GraphNode) = BroadcastedOperator(abs, x)
forward(::BroadcastedOperator{typeof(abs)}, x) = return abs.(x)
backward(node::BroadcastedOperator{typeof(abs)}, x, g) = let
    J = (x.>0) + (x.<0)*(-1)
    tuple(J' * g)
end

#import Base: log
Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x) = return log.(x)
backward(node::BroadcastedOperator{typeof(log)}, x, g) = let
    𝟏 = ones(length(node.output))
    J = 𝟏./x
    #println("log", size(J' * g))
    tuple(J' * g)
end

softmax(x::GraphNode) = BroadcastedOperator(softmax, x)
forward(::BroadcastedOperator{typeof(softmax)}, x) = return exp.(x) ./ sum(exp.(x))
backward(node::BroadcastedOperator{typeof(softmax)}, x, g) = let
    y = node.output
    J = diagm(y) .- y * y'
    tuple(J' * g)
end

function dense(w::GraphNode, b::GraphNode, x::GraphNode, activation::Function) return activation(w * x .+ b) end
function dense(w::GraphNode, x::GraphNode, activation::Function) return activation(w * x) end
function dense(w::GraphNode, x::GraphNode) return w * x end

function mean_squared_loss(y::GraphNode, ŷ::GraphNode)
    n = length(y.output)
    return Constant(1/n) .* sum((y .- ŷ) .^ Constant(2))
end

function cross_entropy(y::GraphNode, ŷ::GraphNode)
    n = length(y.output)
    return Constant(-1/n) .* sum(y .* log.(ŷ))
end

function ReLu(x::GraphNode) return max.(x, Constant(1e-7)) end

#import Base: size
size(x::Variable) = size(x.output)
size(x::Variable, n::Int) = size(x.output, n)

# pooling work

#import Base: getindex
#getindex(A::Variable, r1::UnitRange{Int64}, r2::UnitRange{Int64}, n::Int64) = getindex(Float64, r1, r2, n)

function pooling!(x, kernel::Int=2, stride::Int=2)
    channels = size(x,3)
    output_shape = (size(x,1) - kernel) / stride + 1
    output_shape = round(Int, output_shape)
    output = zeros(output_shape, output_shape, channels)
    @inbounds for channel in 1:channels
        @inbounds for i in 1:stride:size(x,1)
            @inbounds for j in 1:stride:size(x,2)
                size_1 = min(size(x, 1),i+kernel-1)
                size_2 = min(size(x, 2),j+kernel-1)
                output[fld(i,stride)+1, fld(j,stride)+1, channel] = maximum(x[i:size_1,j:size_2,channel])
            end
        end
    end
    return output
end

function conv(x, kernel::Array{Float64}, method::String="valid")
    if size(kernel, 3) == 1
        k_height, k_width = size(kernel)
        k_channels = 1
    else
        k_height, k_width, k_channels = size(kernel)
    end
    height, width = size(x)
    if method == "valid"
        start_h, finish_h = (1, height - k_height + 1)
        start_w, finish_w = (1, width - k_width + 1)
        output = zeros(finish_h, finish_w) 
        @inbounds for i in 1:finish_h
            @inbounds for j in 1:finish_w
                y = 0
                @inbounds for ik in 1:k_width
                    @inbounds for jk in 1:k_height
                        y += x[i + ik - 1, j + jk - 1] * kernel[ik, jk]
                    end
                end
                output[i, j] += y
            end
        end
    elseif method == "full"
        start_h, finish_h = (1 - (k_height - 1), height)
        start_w, finish_w = (1 - (k_width - 1), width)
        output = zeros(height + k_height - 1, width + k_width - 1)
        @inbounds for i in range(start_h, finish_h)
            @inbounds for j in start_w:finish_w
                y = 0
                @inbounds for ik in 1:k_width
                    @inbounds for jk in 1:k_height
                        if (((i + ik - 1) > 0) && ((j + jk - 1) > 0) && ((i + ik - 1) < height + 1) && ((j + jk - 1) < width + 1))
                            y += x[i + ik - 1, j + jk - 1] * kernel[ik, jk]
                        end
                    end
                end
                output[i + k_height - 1, j + k_width - 1] += y
            end
        end
    end
    
    return output
end

function convolution!(x, kernels::Array{Float64}, method::String="valid")
    if size(x, 3) > 1
        i_height, i_width, i_channels = size(x)
    else
        i_height, i_width = size(x)
        i_channels = 1
    end
    k_height = size(kernels,1)
    k_width = size(kernels,2)
    num_of_kernels = last(size(kernels))
    o_height = abs(i_height - k_height) + 1
    o_width = abs(i_width - k_width) + 1
    output = zeros(o_width,o_height,num_of_kernels)

    if i_channels == 1
        @inbounds for kernel in 1:last(size(kernels))
            @inbounds for channel in 1:i_channels
                output[:,:,kernel] += conv(x, kernels[:,:,channel,kernel], method)
            end
        end
    else
        @inbounds for kernel in 1:last(size(kernels))
            @inbounds for channel in 1:i_channels
                output[:,:,kernel] += conv(x[:,:,channel], kernels[:,:,channel,kernel], method)
            end
        end
    end
    output = max.(output, 1e-7)
    return output
end

pooling(x::GraphNode) = BroadcastedOperator(pooling, x)
forward(::BroadcastedOperator{typeof(pooling)}, x) = return pooling!(x)
backward(::BroadcastedOperator{typeof(pooling)}, x, g) = let
    channels = size(x,3)
    output_shape = size(x,1)
    Jx = zeros(size(x))
    output_flat = vec(Jx)
    kernel = 2
    stride = 2
    @inbounds for channel in 1:channels
        @inbounds for i in 1:stride:size(x,1)
            @inbounds for j in 1:stride:size(x,2)
                size_1 = min(size(x, 1),i+kernel-1)
                size_2 = min(size(x, 2),j+kernel-1)
               local_max, max_location = findmax(x[i:size_1,j:size_2, channel])
               Jx[max_location[1]+i-1,max_location[2]+j-1,channel] = g[fld(i,stride)+1,fld(j,stride)+1,channel]
           end
       end
    end
    tuple(Jx)
end

convolution(x::GraphNode, kernels::GraphNode) = BroadcastedOperator(convolution, x, kernels)
forward(::BroadcastedOperator{typeof(convolution)}, x, kernels) = return convolution!(x, kernels)
backward(::BroadcastedOperator{typeof(convolution)}, x, kernels, g) = let
    # prepare empty arrays for holding gradients and roated kernel
    if size(x, 3) > 1
        i_height, i_width, i_channels = size(x)
        Jx = zeros(i_height, i_width, i_channels)
    else
        i_height, i_width = size(x)
        i_channels = 1
        Jx = zeros(i_height, i_width)
    end

    k_height, k_width, k_depth, k_num = size(kernels)
    Jk = zeros(k_height, k_width, k_depth, k_num)

    rotated_kernels = zeros(k_height, k_width, k_depth, k_num)
    @inbounds for l in 1:k_num
        @inbounds for d in 1:k_depth
            @inbounds for i in 1:k_width
                @inbounds for j in 1:k_height
                    rotated_kernels[i,j,d,l] = kernels[k_height - i + 1, k_width - j + 1, d, l]
                end
            end
        end
    end

    # compute gradients
    @inbounds for l in 1:i_channels
        @inbounds for d in 1:k_depth
            Jk[:,:,d,l] = conv(x[:,:,d], g[:,:,l], "valid")
        end
    end

    @inbounds for kernel in 1:k_num
        @inbounds for layer in 1:k_depth
            Jx[:,:,layer] += conv(g[:,:,layer], rotated_kernels[:,:,layer,kernel], "full")
        end
    end
    Jx = isless.(0, Jx)
    tuple(Jx, Jk)
end

flatten(x::GraphNode) = BroadcastedOperator(flatten, x)
forward(::BroadcastedOperator{typeof(flatten)}, x) = return reshape(x, length(x))
backward(::BroadcastedOperator{typeof(flatten)}, x, g) = let
    tuple(reshape(g, size(x)))
end

function prepare_weights(in_channels::Int64, out_channels::Int64, kernel_size::Int64)
    fan_in = in_channels * kernel_size^2
    fan_out = out_channels * kernel_size^2
    variance = 2 / (fan_in + fan_out)
    return randn(Float64, kernel_size, kernel_size, in_channels, out_channels) .* sqrt(variance)
end

function prepare_weights(out_channels::Int64, in_channels::Int64)
    variance = 2 / (in_channels + out_channels)
    return randn(out_channels, in_channels) .* sqrt(variance)
end
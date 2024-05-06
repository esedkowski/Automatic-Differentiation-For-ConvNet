#####################################
#               IMPORT              #
#####################################

using MLDatasets, Images, ImageMagick, Shuffle, LinearAlgebra, Statistics

import Base: +, -, *, /, ^, sin, sum, abs, max, log, size
import Base: show, summary
import LinearAlgebra: mul!

#####################################
#           CORE FUNCTIONS          #
#####################################

abstract type GraphNode end

abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
    name :: String
    Variable(output; name="?") = new(output, nothing, name)
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    name :: String
    BroadcastedOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

#import Base: show, summary
show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op ", x.name, "(", F, ")");
show(io::IO, x::BroadcastedOperator{F}) where {F} = print(io, "op.", x.name, "(", F, ")");
show(io::IO, x::Constant) = print(io, "const ", x.output)
show(io::IO, x::Variable) = begin
    print(io, "var ", x.name);
    print(io, "\n ┣━ ^ "); summary(io, x.output)
    print(io, "\n ┗━ ∇ ");  summary(io, x.gradient)
end

size(x::Variable) = size(x.output)
size(x::Variable, n::Int) = size(x.output, n)

function visit(node::GraphNode, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end
    
function visit(node::Operator, visited, order)
    if node ∈ visited
    else
        push!(visited, node)
        for input in node.inputs
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
    for node in order
        compute!(node)
        reset!(node)
    end
    
    return last(order).output
end

update!(node::Constant, gradient) = nothing
update!(node::GraphNode, gradient) = if isnothing(node.gradient)
    node.gradient = gradient else node.gradient .+= gradient
end

function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
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
    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = BroadcastedOperator(-, x, y)
forward(::BroadcastedOperator{typeof(-)}, x, y) = return x .- y
backward(::BroadcastedOperator{typeof(-)}, x, y, g) = let
    tuple(g,-g)
end

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = BroadcastedOperator(+, x, y)
forward(::BroadcastedOperator{typeof(+)}, x, y) = return x .+ y
backward(::BroadcastedOperator{typeof(+)}, x, y, g) = let
    tuple(g, g)
end

sum(x::GraphNode) = BroadcastedOperator(sum, x)
forward(::BroadcastedOperator{typeof(sum)}, x) = return sum(x)
backward(::BroadcastedOperator{typeof(sum)}, x, g) = let
    𝟏 = ones(length(x))
    J = 𝟏'
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

    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcasted(^, x::GraphNode, n::GraphNode) = BroadcastedOperator(^, x, n)
forward(::BroadcastedOperator{typeof(^)}, x, n) = return x.^n
backward(node::BroadcastedOperator{typeof(^)}, x, n, g) = let
    𝟏 = ones(length(node.output))
    Jx = diagm(n * x .^ (n-1) .* 𝟏)
    Jy = diagm(log.(abs.(x)) .* x .^ n .* 𝟏)
    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcasted(abs, x::GraphNode) = BroadcastedOperator(abs, x)
forward(::BroadcastedOperator{typeof(abs)}, x) = return abs.(x)
backward(node::BroadcastedOperator{typeof(abs)}, x, g) = let
    J = (x.>0) + (x.<0)*(-1)
    tuple(J' * g)
end

Base.Broadcast.broadcasted(log, x::GraphNode) = BroadcastedOperator(log, x)
forward(::BroadcastedOperator{typeof(log)}, x) = return log.(x)
backward(node::BroadcastedOperator{typeof(log)}, x, g) = let
    𝟏 = ones(length(node.output))
    J = 𝟏./x
    tuple(J' * g)
end

softmax(x::GraphNode) = BroadcastedOperator(softmax, x)
forward(::BroadcastedOperator{typeof(softmax)}, x) = return exp.(x) ./ sum(exp.(x))
backward(node::BroadcastedOperator{typeof(softmax)}, x, g) = let
    y = node.output
    J = diagm(y) .- y * y'
    tuple(J' * g)
end

function dense(w, b, x, activation) return activation(w * x .+ b) end
function dense(w, x, activation) return activation(w * x) end
function dense(w, x) return w * x end

function mean_squared_loss(y, ŷ)
    n = length(y.output)
    return Constant(1/n) .* sum((y .- ŷ) .^ Constant(2))
end

function cross_entropy(y, ŷ)
    n = length(y.output)
    return Constant(-1/n) .* sum(y .* log.(ŷ))
end

function ReLu(x) return max.(x, Constant(1e-7)) end

function pooling!(x, kernel::Int=2, stride::Int=2)
    channels = size(x,3)
    output_shape = (size(x,1) - kernel) / stride + 1
    output_shape = round(Int, output_shape)
    output = zeros(output_shape, output_shape, channels)
    for channel in 1:channels
        for i in 1:stride:size(x,1)
            for j in 1:stride:size(x,2)
                size_1 = min(size(x, 1),i+kernel-1)
                size_2 = min(size(x, 2),j+kernel-1)
                output[fld(i,stride)+1, fld(j,stride)+1, channel] = maximum(x[i:size_1,j:size_2,channel])
            end
        end
    end
    return output
end

function conv(x, kernel, method::String="valid")
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
        for i in 1:finish_h
            for j in 1:finish_w
                y = 0
                for ik in 1:k_width
                    for jk in 1:k_height
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
        for i in range(start_h, finish_h)
            for j in start_w:finish_w
                y = 0
                for ik in 1:k_width
                    for jk in 1:k_height
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

function convolution!(x, kernels, method::String="valid")
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
        for kernel in 1:last(size(kernels))
            for channel in 1:i_channels
                output[:,:,kernel] += conv(x, kernels[:,:,channel,kernel], method)
            end
        end
    else
        for kernel in 1:last(size(kernels))
            for channel in 1:i_channels
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
    for channel in 1:channels
       for i in 1:stride:size(x,1)
           for j in 1:stride:size(x,2)
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
    for l in 1:k_num
        for d in 1:k_depth
            for i in 1:k_width
                for j in 1:k_height
                    rotated_kernels[i,j,d,l] = kernels[k_height - i + 1, k_width - j + 1, d, l]
                end
            end
        end
    end

    for l in 1:i_channels
        for d in 1:k_depth
            Jk[:,:,d,l] = conv(x[:,:,d], g[:,:,l], "valid")
        end
    end

    for kernel in 1:k_num
        for layer in 1:k_depth
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

#####################################
#             DATA LOADER           #
#####################################

function load_train_data(shape::Tuple{Int64, Int64}=(28,28), batch_size::Int64=64)
    raw_images, labels = MNIST.traindata(Float64)
    images = []
    for i in 1:size(raw_images, 3)
        image = raw_images[:,:,i]

        image = imresize(image, shape)

        push!(images, image)
    end
    
    num_samples = length(images)
    num_batches = div(num_samples, batch_size)
    batches = []
    indices = shuffle(1:num_samples)
    
    for i in 1:num_batches
        batch_indices = indices[(i-1)*batch_size + 1:i*batch_size]
        batch_images = [images[j] for j in batch_indices]
        batch_labels = [labels[j] for j in batch_indices]
        push!(batches, (batch_images, batch_labels))
    end

    return batches
end
    
function load_test_data(shape::Tuple{Int64, Int64}=(28,28), batch_size::Int64=64)
    raw_images, labels = MNIST.testdata(Float64)
    images = []
    for i in 1:size(raw_images, 3)
        image = raw_images[:,:,i]        
        push!(images, image)
    end
    
    num_samples = length(images)
    num_batches = div(num_samples, batch_size)
    batches = []
    indices = shuffle(1:num_samples)
    
    for i in 1:num_batches
        batch_indices = indices[(i-1)*batch_size + 1:i*batch_size]
        batch_images = [images[j] for j in batch_indices]
        batch_labels = [labels[k] for k in batch_indices]
        push!(batches, (batch_images, batch_labels))
    end
    
    return batches
end

function prep_data(sample, class_num=10)
    x = Variable(sample[1], name="x")
    
    y_raw = zeros(class_num)
    y_raw[sample[2]+1] = 1
    y = Variable(y_raw, name="y")

    return(x, y)
end

#####################################
#            NET EXAMPLE            #
#####################################

training_data = load_train_data()
test_data = load_test_data()

l1w  = Variable(prepare_weights(84, 576), name="wh")
l2w  = Variable(prepare_weights(10, 84), name="wo")
c1w = Variable(prepare_weights(1, 6, 3), name="c1")
c2w = Variable(prepare_weights(6, 16, 3), name="c2")

function net(x, y, l1w, l2w, c1w, c2w)
    c1 = convolution(x, c1w)
    c1.name = "c1"
    p1 = pooling(c1)
    p1.name = "p1"
    c2 = convolution(p1, c2w)
    c2.name = "c1"
    p2 = pooling(c2)
    p2.name = "p2"
    f1 = flatten(p2)
    f1.name = "f1"
    l1 = dense(l1w, f1, ReLu)
    l1.name = "l1"
    l2 = dense(l2w, l1, softmax)
    l2.name = "l2"
    E = mean_squared_loss(y, l2)
    # E = cross_entropy(y, l2)
    E.name = "loss"

    return (topological_sort(l2), topological_sort(E))
end

function training(training_data, l1w, l2w, c1w, c2w, num_of_epochs=9, lr_raw=0.01, show_batch_loss=true)
    losses = Float64[]
    lr = lr_raw
    num_of_batches = size(training_data)[1]
    num_of_samples = size(training_data[1][1])[1]
    for epoch in 1:num_of_epochs
        println("Epoch: ", epoch)
        epoch_losses = Float64[]
        for batch in 1:num_of_batches
            l1_gradient = zeros(size(l1w.output))
            l2_gradient = zeros(size(l2w.output))
            c1_gradient = zeros(size(c1w.output))
            c2_gradient = zeros(size(c2w.output))
            batch_losses = Float64[]
            for sample in 1:num_of_samples
                sample = (training_data[batch][1][sample], training_data[batch][2][sample])
                x, y = prep_data(sample)
                graph_p, graph_l = net(x, y, l1w, l2w, c1w, c2w)
                currentloss = forward!(graph_l)
                backward!(graph_l)
                
                l1_gradient += l1w.gradient
                l2_gradient += l2w.gradient
                c1_gradient += c1w.gradient
                c2_gradient += c2w.gradient

                push!(epoch_losses, first(currentloss))
                push!(batch_losses, first(currentloss))
            end
            if show_batch_loss && batch % 100 == 0
                println("batch ", batch, " mean loss: ", mean(batch_losses))
            end

            l1w.output -= lr*l1_gradient
            l2w.output -= lr*l2_gradient
            c1w.output -= lr*c1_gradient
            c2w.output -= lr*c2_gradient
        end
        println("epoch mean loss: ", mean(epoch_losses))
        if epoch%3==0
            lr = lr/10
        end
    end
    return (l1w, l2w, c1w, c2w)
end

function testing(test_data, l1w, l2w, c1w, c2w, show_batch_loss=true)
    correct = 0
    all = 0

    num_of_batches = size(test_data)[1]
    num_of_samples = size(test_data[1][1])[1]
    
    for batch in 1:num_of_batches
        batch_correct = 0
        loc_all = 0
        for sample in 1:num_of_samples
            sample = (test_data[batch][1][sample], test_data[batch][2][sample])

            x, y = prep_data(sample)

            graph_p, graph_l = net(x, y, l1w, l2w, c1w, c2w)
            answ = forward!(graph_p)
            
            all += 1
            loc_all += 1
            if findmax(y.output)[2] == findmax(answ)[2]
                batch_correct += 1
                correct += 1
            end
        end
    end
    println("Acc = ", correct/all)
end

@time l1w, l2w, c1w, c2w = training(training_data, l1w, l2w, c1w, c2w)
@time testing(test_data, l1w, l2w, c1w, c2w)
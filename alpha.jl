using LinearAlgebra, Statistics
include("core.jl")
include("dataloader.jl")

training_data = load_train_data() # training_data[batch][imgs(1) or labels(2)][img/label]
test_data = load_test_data() # test_data[batch][imgs(1) or labels(2)][img/label]

# sample = (training_data[1][1][1], training_data[1][2][1])
# size(training_data) - liczba batchy 
# size(training_data[1][1]) - liczba zdjęć w batch'u

l1w  = Variable(prepare_weights(84, 576))
l2w  = Variable(prepare_weights(10, 84))
c1w = Variable(prepare_weights(1, 6, 3))
c2w = Variable(prepare_weights(6, 16, 3))

function net(x, y, l1w, l2w, c1w, c2w)
    c1 = convolution(x, c1w)
    p1 = pooling(c1)
    c2 = convolution(p1, c2w)
    p2 = pooling(c2)
    f1 = flatten(p2)
    l1 = dense(l1w, f1, ReLu)
    l2 = dense(l2w, l1, softmax)
    E = mean_squared_loss(y, l2)
    #E = cross_entropy(y, l2)

    return (topological_sort(l2), topological_sort(E))
end

function training(training_data, l1w, l2w, c1w, c2w, num_of_epochs=3, lr_raw=0.01, show_batch_loss=true)
    losses = Float64[]
    lr = lr_raw
    num_of_batches = size(training_data)[1]
    num_of_samples = size(training_data[1][1])[1]
    @inbounds for epoch in 1:num_of_epochs
        println("Epoch: ", epoch)
        epoch_losses = Float64[]
        @inbounds for batch in 1:num_of_batches
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
    end
    return (l1w, l2w, c1w, c2w)
end

function testing(test_data, l1w, l2w, c1w, c2w, show_batch_loss=true)
    correct = 0
    all = 0

    num_of_batches = size(test_data)[1]
    num_of_samples = size(test_data[1][1])[1]
    
    @inbounds for batch in 1:num_of_batches
        batch_correct = 0
        loc_all = 0
        @inbounds for sample in 1:num_of_samples
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
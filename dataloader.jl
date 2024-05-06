using MLDatasets, Images, ImageMagick, Shuffle

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
    x = Variable(sample[1])
    
    y_raw = zeros(class_num)
    y_raw[sample[2]+1] = 1
    y = Variable(y_raw)

    return(x, y)
end
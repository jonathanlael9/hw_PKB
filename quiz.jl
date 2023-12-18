using Serialization
using DataFrames
using Statistics
using LinearAlgebra

function split_data(data)
    features = data[:, 1:end-1]  # Ambil kolom 1 hingga kolom ke-(akhir-1)
    labels = data[:, end]       # Ambil kolom terakhir sebagai label
    return features, labels
end

function calculate_miu(features, labels)
    unique_labels = unique(labels)
    max_features = size(features, 2)
    miu_data = zeros(Float64, length(unique_labels), max_features)

    for (i, label) in enumerate(unique_labels)
        indices = findall(x -> x == label, labels)
        if isempty(indices)
            # Handle the case where a class has no occurrences
            miu_data[i, :] .= mean(convert.(Float64, features), dims=1)[:]
        else
            miu_data[i, :] .= mean(convert.(Float64, features[indices, :]), dims=1)[:]
        end
    end

    return miu_data
end

function split_miu_data(miu_data)
    miu_columns = [convert(Matrix{Float16}, miu_data[:, i]) for i in 1:size(miu_data, 2)]
    return miu_columns
end


function predict_class(data, miu_columns)
    predicted_class = zeros(Int, size(data, 1))
    data_float16 = convert(Matrix{Float16}, data)
    for i in 1:size(data_float16, 1)
        distances = [norm(data_float16[i, :] .- miu) for miu in miu_columns]
        min_index = argmin(distances)
        predicted_class[i] = min_index[1]  
    end
    return predicted_class
end

function calculate_accuracy(predicted_class, actual_class)
    return sum(predicted_class .== actual_class) / length(actual_class)
end

function display_first_few_rows(matrix, max_rows=5)
    display(DataFrame(matrix[1:min(end, max_rows), :], :auto))
end

function main()
    file_path = "data_9m.mat"
    data = deserialize(file_path)

    println("Data : ")
    display_first_few_rows(data)

    features, labels = split_data(data)

    println("\nFeatures : ")
    display_first_few_rows(features)

    miu_data = calculate_miu(features, labels)

    println("\nMiu Data : ")
    display(DataFrame(miu_data, :auto))

    miu_columns_float16 = convert(Matrix{Float16}, miu_data)
    accuracies = Float64[]

    for col in 1:size(features, 2)
        data_part = hcat(features[:, col], labels)
        predicted_class = predict_class(data_part[:, 1:end-1], miu_columns_float16)
        accuracy = calculate_accuracy(predicted_class, data_part[:, end])
        push!(accuracies, accuracy)
        println("\nAccuracy for Feature $col : $(accuracy * 100)%")
    end
    
end

main()

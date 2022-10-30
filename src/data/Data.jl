__precompile__()

module Data

    export DataModule, train_dataloader, val_dataloader, get_dataloader
    
    using ..HyperParameters
    struct DataModule <: HyperParameter
        root::String
        num_workers::Integer
        function DataModule(root::String="..../data", num_workers::Integer = 4)
            t = new(root, num_workers)
            save_hyperparameters(t)
            return t
        end
    end

    function get_dataloader(data::DataModule, train::Bool)
        error("Not implemented")
    end
   
    function train_dataloader(data::DataModule)
        return get_dataloader(data, true)
    end

    function verify_dataloader(data::DataModule)
        return get_dataloader(data, false)
    end
end
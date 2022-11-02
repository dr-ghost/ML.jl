__precompile__()

module Data

    export DataModule, train_dataloader, val_dataloader, get_dataloader
    
    using ..HyperParameters
    
    abstract type DataModule <: HyperParameter end

    function data_init(data::DataModule, root = "../data", num_workers = 2)
        if hasfield(data, :root)
            data.root = root
        end
        if hasfield(data, :num_workers)
            data.num_workers = num_workers
        end
        save_hyperparameters(data)
    end

    function get_dataloader(data::DataModule, b_train::Bool)
        error("Not implemented")
    end
   
    function train_dataloader(data::DataModule)
        return get_dataloader(data, true)
    end

    function verify_dataloader(data::DataModule)
        return get_dataloader(data, false)
    end
end
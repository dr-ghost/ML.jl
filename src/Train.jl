__precompile__()

module _Train_

    export Trainer , data_prep, prepare_model, fit, configure_optimizers

    using .._HyperParameter_, .._Data_, .._Model_
    using  .._Optimizer_

    struct Trainer <: HyperParameter
        max_epochs::Integer
        num_gpus::Integer
        gradient_clip_val::Integer
        train_dataloader
        val_dataloader
        num_train_batches::Integer
        val_train_batches::Integer
        epoch::Integer
        optimizer::Optimizer        
        function Trainer(max_epochs::Integer = 100, num_gpus::Integer = 0, gradient_clip_val::Integer = 0, optimizer::Optimizer = nothing)
            @assert num_gpus == 0 "No gpu sumport"
            t = new(max_epochs, num_gpus, gradient_clip_val, optimizer)
            save_hyperparameters(t)
            return t
        end
    end

    function data_prep(trainer::Trainer, data::DataModule)
        trainer.train_dataloader = train_dataloader(data)
        trainer.val_dataloader = val_dataloader(data)

        trainer.num_train_batches = length(train_dataloader)

        trainer.val_train_batches = if val_dataloader !== nothing length(val_dataloader) else 0 end

    end

    function prepare_model(model::Model, trainer::Trainer)
        Model.progress_board.xLim = (0, trainer.max_epochs)
    end

    function configure_optimizers(trainer::Trainer)
        error("Not implemented")
    end

    function fit(model::Model, trainer::Trainer, data::DataModule)
        data_prep(trainer, data)
        prepare_model(Model, trainer)
        configure_optimizers(trainer)

        trainer.epoch = 0

        for trainer.epoch = 1:trainer.max_epochs
            epoch_fit(Model, trainer)
        end
    end

    function epoch_fit(model::Model, trainer::Trainer)
        error("Not implemented")
    end

end
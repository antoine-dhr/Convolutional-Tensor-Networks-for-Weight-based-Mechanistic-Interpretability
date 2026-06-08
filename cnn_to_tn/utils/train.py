import torch
from utils.wandbai import initialize_wandb_run
import wandb
import gc
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torchvision.transforms import v2

def evaluate_model(model, data_loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            _, top5_pred = torch.topk(outputs, 5, dim=1)

            total += labels.size(0)
            top1_correct += (predicted == labels).sum().item()
            top5_correct += top5_pred.eq(labels.view(-1, 1)).sum().item()

    average_loss = total_loss / len(data_loader.dataset)
    top1_accuracy = 100 * top1_correct / total
    top5_accuracy = 100 * top5_correct / total

    return average_loss, top1_accuracy, top5_accuracy

def train_model(model, 
                train_dataset, 
                test_dataset, 
                config, 
                optimizer_arg, 
                criterion_arg, 
                # Is this method used to tune?
                tuning_flag,
                project_name=None,
                run_name=None,
                apply_mixup=True):
    
    if wandb.run is None:
        initialize_wandb_run(project_name=project_name, run_name=run_name, config=config)

    load_pretrained_model = bool(getattr(config, "load_pretrained_model", False))

    if load_pretrained_model:
        configured_artifact_name = getattr(config, "pretrained_model_artifact", None)
        artifact_name = str(configured_artifact_name).strip() if configured_artifact_name is not None else ""
        if artifact_name == "":
            artifact_name = "model-weights:latest"
        elif ":" not in artifact_name:
            artifact_name = f"{artifact_name}:latest"

        if artifact_name.endswith(":latest"):
            print(
                "Warning: loading ':latest' artifact alias. "
                "For fully reproducible fine-tuning, pin a versioned artifact name."
            )

        load_model_weights(model=model, device="cpu", artifact_name=artifact_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    use_torch_compile = bool(getattr(config, "use_torch_compile", False))
    torch_compile_mode = str(getattr(config, "torch_compile_mode", "default"))

    model.to(device)
    if device == "cuda" and use_torch_compile:
        try:
            model = torch.compile(model, mode=torch_compile_mode, fullgraph=False)
            print(f"Torch compile enabled with mode='{torch_compile_mode}'")
        except Exception as exc:
            print(f"Torch compile failed ({exc}). Continuing without compile.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.run.summary["total_parameters"] = total_params
    wandb.run.summary["trainable_parameters"] = trainable_params
    print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

    criterion = criterion_arg(label_smoothing=config.label_smoothing).to(device)

    learning_rate = config.learning_rate 
    weight_decay = config.weight_decay
    batch_size = config.batch_size
    num_epochs = config.epochs
    optimizer_name = str(getattr(config, "optimizer", "")).lower()

    optimizer_kwargs = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }
    if optimizer_name == "sgd" and hasattr(config, "momentum"):
        optimizer_kwargs["momentum"] = config.momentum

    optimizer = optimizer_arg(model.parameters(), **optimizer_kwargs)

    scaler = torch.amp.GradScaler(device='cuda', enabled=use_amp)

    warmup_epochs = config.warmup_epochs

    if warmup_epochs > 0:
        # From ~0 to start LR over warmup_epochs
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs)
        # Cosine annealing over the remaining epochs after warmup
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
        # Chain both schedulers
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    else:
        # No warmup: pure cosine annealing from the start
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Train data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # Validation data or testing data (depending on tuning_flag)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    num_classes = model.fc.out_features
    # Randomly apply CutMix / MixUp to each training batch (configurable)
    mixup_alpha = float(getattr(config, "mixup_alpha", 0.2))
    cutmix_alpha = float(getattr(config, "cutmix_alpha", 1.0))
    use_mixup = bool(getattr(config, "use_mixup", True))
    use_cutmix = bool(getattr(config, "use_cutmix", True))

    if apply_mixup and (use_mixup or use_cutmix):
        batch_transforms = []
        if use_cutmix:
            batch_transforms.append(v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha))
        if use_mixup:
            batch_transforms.append(v2.MixUp(num_classes=num_classes, alpha=mixup_alpha))

        if len(batch_transforms) == 1:
            cutmix_or_mixup = batch_transforms[0]
        else:
            cutmix_or_mixup = v2.RandomChoice(batch_transforms)
    else:
        cutmix_or_mixup = None

    evaluate_before_training = bool(
        getattr(config, "evaluate_before_training", getattr(config, "load_pretrained_model", False))
    )

    if evaluate_before_training:
        # Evaluate loaded model at its current fine-tuning/pre-training RMS configuration.
        baseline_loss, baseline_top1, baseline_top5 = evaluate_model(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        wandb.log(
            {
                "PretrainLoad/Loss": baseline_loss,
                "PretrainLoad/Top-1 accuracy": baseline_top1,
                "PretrainLoad/Top-5 accuracy": baseline_top5,
            }
        )
        print(
            f"Pre-training baseline | Loss: {baseline_loss:.4f}, "
            f"Top-1: {baseline_top1:.2f}%, Top-5: {baseline_top5:.2f}%"
        )

    for epoch in range(num_epochs):
        # TRAINING
        model.train()
        training_loss = 0.0
        train_correct = 0
        train_top5_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply CutMix or MixUp (labels become one-hot soft targets)
            if cutmix_or_mixup is not None:
                inputs, labels = cutmix_or_mixup(inputs, labels)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            training_loss += loss.item() * inputs.size(0)

            if cutmix_or_mixup is not None:
                # Labels are soft (one-hot mixed), so compare against the dominant class
                true_class = labels.argmax(dim=1)
            else:
                true_class = labels

            _, predicted = torch.max(outputs, 1)
            _, top5_pred = torch.topk(outputs, 5, dim=1)

            train_total += true_class.size(0)
            train_correct += (predicted == true_class).sum().item()
            train_top5_correct += top5_pred.eq(true_class.view(-1, 1)).sum().item()

        epoch_training_loss = training_loss / len(train_loader.dataset)
        train_accuracy = 100 * train_correct / train_total
        train_top5_accuracy = 100 * train_top5_correct / train_total

        lr_scheduler.step()

        # EVALUATION
        epoch_validation_loss, evaluation_accuracy, evaluation_top5_accuracy = evaluate_model(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        current_lr  = optimizer.param_groups[0]['lr']

        log_dict = {
            "epoch": epoch + 1,
            ("Loss/validation" if tuning_flag else "Loss/testing"): epoch_validation_loss,
            "Loss/train": epoch_training_loss,
            "Top-1 accuracy/training": train_accuracy,
            ("Top-1 accuracy/validation" if tuning_flag else "Top-1 accuracy/testing"): evaluation_accuracy,
            "Top-5 accuracy/training": train_top5_accuracy,
            ("Top-5 accuracy/validation" if tuning_flag else "Top-5 accuracy/testing"): evaluation_top5_accuracy,
            "Optimizer/learning_rate": current_lr,
        }

        wandb.log(log_dict, step=epoch)

    print(f"{'Tuning' if tuning_flag else 'Testing'} complete. Cleaning up...")
    
    # Save model weights to wandb
    model_weights = model.state_dict()
    torch.save(model_weights, "tmp_model_weights.pt")
    artifact = wandb.Artifact("model-weights", type="model")
    artifact.add_file("tmp_model_weights.pt")
    wandb.log_artifact(artifact)
    os.remove("tmp_model_weights.pt")
    print("Model weights saved to wandb")
    
    # Save full checkpoint for resuming training to wandb
    config_dict = vars(config) if not isinstance(config, dict) else config
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "config": config_dict,
        "epoch": num_epochs - 1
    }
    torch.save(checkpoint, "tmp_checkpoint.pt")
    artifact = wandb.Artifact("training-checkpoint", type="checkpoint")
    artifact.add_file("tmp_checkpoint.pt")
    wandb.log_artifact(artifact)
    os.remove("tmp_checkpoint.pt")
    print("Training checkpoint saved to wandb")
    
    # Finish the W&B process explicitly
    wandb.finish()

    # Move model to CPU and delete to free VRAM
    model.cpu()
    del model
    del optimizer
    
    # Clear the cache so the next sweep run starts with 0MB used
    torch.cuda.empty_cache()
    gc.collect() 

    return None


def load_model_weights(model, run_name=None, device="cuda", artifact_name="model-weights:latest"):
    artifact = wandb.use_artifact(artifact_name)
    artifact_path = artifact.download()
    model_path = os.path.join(artifact_path, "tmp_model_weights.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    resolved_name = getattr(artifact, "name", artifact_name)
    resolved_version = getattr(artifact, "version", "unknown")
    resolved_digest = getattr(artifact, "digest", "unknown")
    wandb.run.summary["pretrained_artifact_requested"] = artifact_name
    wandb.run.summary["pretrained_artifact_resolved"] = resolved_name
    wandb.run.summary["pretrained_artifact_version"] = resolved_version
    wandb.run.summary["pretrained_artifact_digest"] = resolved_digest
    print(
        f"Model loaded from wandb artifact '{resolved_name}' "
        f"(requested: '{artifact_name}', version: {resolved_version}, digest: {resolved_digest})"
    )
    return model

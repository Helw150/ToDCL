import torch
from tqdm import tqdm

def update_adapters(task_name, task_loader, model):
    if torch.cuda.is_available():
        model.cuda()

    new_adapter_weights = find_best_merge(task_name, task_loader, model)
    # Modify the weights of the adapters based on what is returned
    active_state_dict = model.state_dict()
    task_id = str(model.task_list_seen.index(task_name))

    # Change the current model to have adapter weights that were
    # selected by [find_best_merge]
    for layer_name, new_tensor_weight in new_adapter_weights.items():
        assert active_state_dict.get(layer_name) is not None
        active_state_dict[layer_name] = new_tensor_weight

    model.load_state_dict(active_state_dict)
    return model


def find_best_merge(current_task_name, task_loader, model):
    # I am expecting best weights to simply be
    # a dictionary of named layer to the tensor of weights/bias
    # The named layer parameter e.g. ("model.encoder.block.0.layer.0.adapters.3.adapter_up.weight" : weight tensor)
    #
    # This should be able to be found when merging as in order to merge you will probably
    # want to iterate through model.parameters() which will give you name,tensor
    # In the code they check if ["adapter" in layer_name] to determine
    # if this is an adapter layer

    best_score = float('-inf')
    best_weights = None

    current_task_id = str(model.task_list_seen.index(current_task_name))
    # Iterate Through and Find Best Merge
    for task_name in tqdm(model.task_list_seen):
        task_id = str(model.task_list_seen.index(task_name))
        score, weights = score_merge(current_task_id, task_id, task_loader, model)
        if score > best_score:
            best_score = score
            best_weights = weights
        if best_weights is None:
            best_weights = weights

    return best_weights

def name_parser(task_id, original_name):
    # Experimentally, the task id seems to be put at index 7
    parts = original_name.split(".")
    assert parts[7] == task_id
    prefix = ".".join(parts[:7])
    suffix = ".".join(parts[8:])
    return prefix, suffix

def score_merge(current_task_id, task_id, task_loader, model):
    score = None

    fisher_target, param_target = compute_fisher(task_loader, model, current_task_id)
    fisher_source, param_source = compute_fisher(task_loader, model, task_id)

    best_score = float("-inf")
    best_weights = None
    for lamb in tqdm(torch.linspace(0.1, 1, 10)):
        tqdm.write(f'Trying lambda = {lamb}')
        weights = set_params(
            model,
            fisher_source,
            param_source,
            fisher_target,
            param_target,
            lamb,
            current_task_id,
        )

        score = evaluate(task_loader, model)
        tqdm.write(f"Calculated score = {score} with lambda = {lamb}")
        if score > best_score:
            best_score = score
            best_weights = weights
        if best_weights is None:
            best_weights = weights

    return best_score, best_weights


def evaluate(task_loader, model):
    model.eval()

    loss = 0
    for idx, inputs in enumerate(task_loader):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if torch.cuda.is_available():
                    inputs[k] = v.cuda()

        current_output = model.model(
            input_ids=inputs["encoder_input"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["decoder_output"],
        )
        current_loss = current_output.loss
        loss -= current_loss

    if torch.isnan(torch.tensor(loss)):
        loss = float('-inf')

    return loss


def set_params(
    model, fisher_source, param_source, fisher_target, param_target, lamb, target_id
):
    param_dict = {}
    lamb_2 = 1 - lamb
    model.model.train_adapter("temporary")
    model.model.set_active_adapters("temporary")
    for name, param in model.named_parameters():
        if param.requires_grad:
            prefix, suffix = name_parser("temporary", original_name=name)
            name = prefix + suffix
            source_fish = fisher_source[name] / torch.norm(fisher_source[name])
            target_fish = fisher_target[name] / torch.norm(fisher_target[name])

            # Fisher Flooring
            source_fish[source_fish <= 1e-6] = 1e-6
            target_fish[target_fish <= 1e-6] = 1e-6

            source = param_source[name]
            target = param_target[name]

            original_min = torch.min(target)
            original_max = torch.max(target)

            # Default to Target if Fisher is small
            normalization_factor = (lamb * source_fish + lamb_2 * target_fish)
            merge = (
                (lamb * source_fish * source) + (lamb_2 * target_fish * target)
            ) / normalization_factor
            merge = torch.clamp(merge, original_min, original_max)
            param.data.copy_(merge)
            param_dict[prefix + "." + target_id + "." + suffix] = merge
    return param_dict


def compute_fisher(task_loader, model, task_id, num_samples=1024):

    model.model.train_adapter(task_id)
    model.model.set_active_adapters(task_id)
    gradients_dict = {}
    param_dict = {}

    for name, param in model.named_parameters():
        # Only Compute For Active Adapter
        if param.requires_grad:
            prefix, suffix = name_parser(task_id, name)
            name = prefix + suffix
            param_dict[name] = param
            gradients_dict[name] = torch.zeros_like(param)

    for idx, inputs in enumerate(task_loader):
        if idx >= num_samples:
            break

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if torch.cuda.is_available():
                    inputs[k] = v.cuda()

        loss = model.model(
            input_ids=inputs["encoder_input"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["decoder_output"],
        )[0]

        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                prefix, suffix = name_parser(task_id, name)
                name = prefix + suffix
                gradients_dict[name] += torch.square(param.grad).data

        model.zero_grad()

    return gradients_dict, param_dict

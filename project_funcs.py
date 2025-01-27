import torch
from tqdm import tqdm


def update_adapters(task_name, task_loader, model):
    if torch.cuda.is_available():
        model.cuda()

    new_adapter_weights = find_best_merge(task_name, task_loader, model)
    # Modify the weights of the adapters based on what is returned
    active_state_dict = model.state_dict()
    task_id = model.adapters[model.task_list_seen.index(task_name)]

    # Change the current model to have adapter weights that were
    # selected by [find_best_merge]
    for layer_name, new_tensor_weight in new_adapter_weights.items():
        if active_state_dict.get(layer_name) is None:
            layer_name = get_unique_name(task_id, layer_name)
        if active_state_dict.get(layer_name) is None:
            print(f"Tried to insert {layer_name} but no such layer existed")
            all_str = "\n\t".join(active_state_dict.keys())
            print(f"Existing layers are: \n\t{all_str}")
            raise LookupError(f"Impossible state reached: can't insert {layer_name}")

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

    best_score = float("-inf")
    best_weights = None

    current_task_id = model.adapters[model.task_list_seen.index(current_task_name)]
    # Iterate Through and Find Best Merge
    for task_name in tqdm(model.task_list_seen):
        task_id = model.adapters[model.task_list_seen.index(task_name)]
        score, weights = score_merge(current_task_id, task_id, task_loader, model)
        if score > best_score:
            best_score = score
            best_weights = weights
        if best_weights is None:
            best_weights = weights

    return best_weights


def name_parser(task_id, original_name):
    # Experimentally, the task id seems to be put at index 7
    parts = original_name.split(f".{task_id}.")
    assert len(parts) == 2
    prefix = parts[0]
    suffix = parts[1]
    return prefix, suffix


def get_universal_name(task_id, original_name):
    prefix, suffix = name_parser(task_id, original_name)
    return prefix + ".UNIVERSAL_TASK_ADAPTER." + suffix


def get_unique_name(task_id, original_name):
    prefix, suffix = name_parser("UNIVERSAL_TASK_ADAPTER", original_name)
    return prefix + f".{task_id}." + suffix


def score_merge(current_task_id, task_id, task_loader, model):
    score = None

    fisher_target, param_target = compute_fisher(task_loader, model, current_task_id)
    fisher_source, param_source = compute_fisher(task_loader, model, task_id)

    model.model.set_active_adapters(current_task_id)
    score, nan_percentage = evaluate(task_loader, model, return_nan_percentage=True)
    best_score = score
    best_weights = param_target
    tqdm.write(
        f"Starting ({current_task_id}): Calculated score = {score}  and NAN % = {nan_percentage}"
    )
    for lamb in tqdm(torch.linspace(0.1, 1, 10)):

        assert_same = current_task_id == task_id
        weights = set_params(
            model,
            fisher_source,
            param_source,
            fisher_target,
            param_target,
            lamb,
            current_task_id,
            assert_same=assert_same,
        )

        score, nan_percentage = evaluate(task_loader, model, return_nan_percentage=True)
        tqdm.write(
            f"Merging ({task_id} -> {current_task_id}): Calculated score = {score} with lambda = {lamb} and NAN % = {nan_percentage}"
        )
        if score > best_score:
            best_score = score
            best_weights = weights

    return best_score, best_weights


def evaluate(task_loader, model, return_nan_percentage=False):
    model.eval()

    nan_counter = 0
    total_counter = 0
    loss = 0
    for idx, inputs in enumerate(task_loader):

        # Free unused GPU Memory
        torch.cuda.empty_cache()

        total_counter += 1
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if torch.cuda.is_available():
                    inputs[k] = v.cuda()

        current_loss = model.model(
            input_ids=inputs["encoder_input"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["decoder_output"],
        ).loss

        if torch.isnan(torch.tensor(current_loss)):
            nan_counter += 1
            current_loss = 0

        # This detaches the gradient
        loss -= float(current_loss)

        del current_loss

    # Skip NANS if less than 5% NAN
    nan_percentage = nan_counter / total_counter
    if nan_percentage >= 0.05:
        loss = float("-inf")

    if return_nan_percentage:
        return loss, nan_percentage
    return loss


def set_params(
    model,
    fisher_source,
    param_source,
    fisher_target,
    param_target,
    lamb,
    target_id,
    assert_same=False,
    ignore_fisher_cutoff=1e-6,
):
    param_dict = {}
    lamb_2 = 1 - lamb
    model.model.train_adapter("temporary")
    model.model.set_active_adapters("temporary")
    for name, param in model.named_parameters():
        if param.requires_grad:
            name = get_universal_name("temporary", original_name=name)
            fisher_source_norm = torch.linalg.norm(fisher_source[name])
            fisher_target_norm = torch.linalg.norm(fisher_target[name])

            source = param_source[name]
            target = param_target[name]

            if (
                fisher_source_norm <= ignore_fisher_cutoff
                or fisher_target_norm <= ignore_fisher_cutoff
            ):
                # Theres no point in merging because our fisher matrices
                # are just 0.
                param.data.copy_(target)
                param_dict[get_unique_name(target_id, name)] = target
                continue

            source_fish = fisher_source[name] / fisher_source_norm
            target_fish = fisher_target[name] / fisher_target_norm

            # Fisher Flooring
            source_fish[source_fish <= 1e-6] = 1e-6
            target_fish[target_fish <= 1e-6] = 1e-6

            # Default to Target if Fisher is small
            normalization_factor = lamb * source_fish + lamb_2 * target_fish
            merge = (
                (lamb * source_fish * source) + (lamb_2 * target_fish * target)
            ) / normalization_factor

            if assert_same and (not torch.allclose(target, merge)):
                tqdm.write("Model merging with self was not an identity operation!")
                tqdm.write(f"Original Params: {target}")
                tqdm.write(f"Raw Target Fisher: {fisher_target[name]}")
                tqdm.write(f"Normalized Target Fisher: {target_fish}")
                tqdm.write(f"Source Params: {source}")
                tqdm.write(f"Raw Source Fisher: {fisher_source[name]}")
                tqdm.write(f"Normalized Source Fisher: {source_fish}")
                tqdm.write(f"Merged Params: {merge}")
                raise ArithmeticError("Merging operation did not perform as expected.")

            param.data.copy_(merge)
            param_dict[get_unique_name(target_id, name)] = merge
    return param_dict


def compute_fisher(task_loader, model, task_id, num_samples=1024):
    torch.manual_seed(50)
    model.model.train_adapter(task_id)
    model.model.set_active_adapters(task_id)
    gradients_dict = {}
    param_dict = {}

    for name, param in model.named_parameters():
        # Only Compute For Active Adapter
        if param.requires_grad:
            name = get_universal_name(task_id, name)
            param_dict[name] = param
            gradients_dict[name] = torch.zeros_like(param)

    for idx, inputs in enumerate(task_loader):

        model.zero_grad()

        # Free unused GPU Memory
        torch.cuda.empty_cache()

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
                name = get_universal_name(task_id, name)
                gradients_dict[name] += torch.square(param.grad).data

        # Freeing some GPU references
        del loss

    model.zero_grad()

    return gradients_dict, param_dict

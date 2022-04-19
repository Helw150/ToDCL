def update_adapters(task_id, task_loader, model):
    new_adapter_weights = find_best_merge(task_id, task_loader, model)
    # Modify the weights of the adapters based on what is returned
    active_state_dict = model.state_dict()

    return model


def find_best_merge(current_task_id, task_loader, model):
    # I am expecting [best_weights] to be an ordered list of
    # tensors [t1, t2, t3, ...] where each of t_i are the weights
    # of the 1st, 2nd, ... adapter.
    best_weights = None

    # Iterate Through and Find Best Merge
    for task_id in model.task_list_seen:
        score, weights = score_merge(current_task_id, task_id, task_loader, model)

    return best_weights


def score_merge(current_task_id, task_id, task_loader, model):
    score = None
    weights = torch.zeros(100)

    for lamb in range(0, 1, 0.1):
        lamb_2 = 1 - lamb
        # Merge and Score Here

    return score, weights

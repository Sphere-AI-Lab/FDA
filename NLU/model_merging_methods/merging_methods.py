from collections import defaultdict, OrderedDict
from tqdm import tqdm
import copy
import types
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_merging_methods.task_vector import TaskVector
from utils.utils import get_param_names_to_merge, get_modules_to_merge
from model_merging_methods.distill_merging_utils import *


device = 'cuda:0'


class MergingMethod:
    def __init__(self, merging_method_name: str, language_model_name: str):
        """
        Methods for model merging.
        :param merging_method_name: str, name of the merging method, can be "average_merging", "task_arithmetic",
        "fisher_merging", "regmean_merging", "ties_merging", "latent_merging"
        :return:
        """
        self.merging_method_name = merging_method_name
        self.language_model_name = language_model_name

    def copy_params_to_model(self, params: dict, model: nn.Module):
        """
        copy parameters in "params" to the model
        :param params: dict, dictionary of parameters
        :param model: nn.Module, model that needs to copy parameters
        :return:
        """
        for param_name, param_value in model.named_parameters():
            if param_name in params:
                param_value.data.copy_(params[param_name])

    def average_merging(self, models_to_merge: list, exclude_param_names_regex: list):
        """
        average merging method
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :return:
        """
        # dictionary of list, where key is the parameter name,
        # value is a list of the corresponding parameters of all the models that need to be merged
        models_to_merge_param_dict = defaultdict(list)
        # iterate each individual model that needs to be merged
        for model_to_merge in models_to_merge:
            param_dict = {param_name: param_value for param_name,
                          param_value in model_to_merge.named_parameters()}
            # exclude parameter whose name matches element in exclude_param_names_regex
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(
                param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            for param_name in param_names_to_merge:
                models_to_merge_param_dict[param_name].append(
                    param_dict[param_name])

        with torch.no_grad():
            # average merging of individual models' parameters
            averaged_params = {param_name: torch.stack(model_to_merge_param, dim=0).mean(
                dim=0) for param_name, model_to_merge_param in models_to_merge_param_dict.items()}

        return averaged_params

    def task_arithmetic(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, scaling_coefficient: float = 1.0):
        """
        task arithmetic method
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :return:
        """
        assert isinstance(
            scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

        models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge,
                                                   exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

        # for task_vector in models_to_merge_task_vectors:
        #     print(task_vector.norm())
        # iterate each individual model that needs to be merged
        with torch.no_grad():
            # sum up the task vectors
            for index in range(len(models_to_merge_task_vectors)):
                print(models_to_merge_task_vectors[index].norm())
            merged_task_vector = models_to_merge_task_vectors[0] + \
                models_to_merge_task_vectors[1]
            for index in range(2, len(models_to_merge_task_vectors)):
                merged_task_vector = merged_task_vector + \
                    models_to_merge_task_vectors[index]

            # combine with parameters of the merged model based on scaling coefficient
            merged_params = merged_task_vector.combine_with_pretrained_model(
                pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

        return merged_params
    
    def adamerging(
        self,
        merged_model: nn.Module,
        models_to_merge: list,
        trainers: list,
        exclude_param_names_regex: list,
        lr=0.01,
        epochs=1,
        granularity="layerwise",
    ):
        classifier_model = []
        for model in models_to_merge:
            classifier_model.append(copy.deepcopy(model.classifier).to(device))

        if self.language_model_name == 'bert-base-uncased':
            new_forward = new_forward_bert
        elif self.language_model_name == 'roberta-base':
            new_forward = new_forward_roberta
        else:
            raise NotImplementedError(
                f'Language model {self.language_model_name} is not supported')

        for i in range(len(models_to_merge)):
            models_to_merge[i].forward = types.MethodType(
                new_forward, models_to_merge[i])
        merged_model.forward = types.MethodType(new_forward, merged_model)

        for i in range(len(models_to_merge)):
            models_to_merge[i] = del_ex(
                models_to_merge[i], exclude_param_names_regex)
        # merged_model = del_ex(merged_model, exclude_param_names_regex)
        merged_model = del_ex(merged_model, exclude_param_names_regex).cuda()

        all_combined_model = MergedModel(merged_model, models_to_merge, granularity,0.3).to(device)

        optimizer = torch.optim.Adam(all_combined_model.parameters(), lr=lr)

        combined_loader = merge_data_loaders_from_trainers(trainers)

        print('Start training')
        for epoch in range(epochs):
            all_combined_model.train()
            for data in tqdm(combined_loader):
                x = data['data'][0].to(device)
                source_loader = data['source_loader'].to(device)
                optimizer.zero_grad()

                feature = all_combined_model(x)

                # Calculate loss
                loss = 0
                for idx in range(len(models_to_merge)):
                    logits = classifier_model[idx](feature)
                    loss += (softmax_entropy(logits) *
                                source_loader.eq(idx).float()).sum()

                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs} finished')

        merged_params = all_combined_model.get_named_parameters()

        return merged_params
    

    def fisher_merging(self, models_to_merge: list, trainers: list, exclude_param_names_regex: list, nums_fisher_examples: list, fisher_scaling_coefficients: list = None,
                       normalize_fisher_weight: bool = True, minimal_fisher_weight: float = 1e-6):
        """
        fisher merging method
        :param models_to_merge: list, individual models that need to be merged
        :param trainers: list, trainers of individual models
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param nums_fisher_examples: list, numbers of examples to compute fisher weights
        :param fisher_scaling_coefficients: list, scaling coefficients to merge fisher weights
        :param normalize_fisher_weight: boolean, whether to normalize fisher weights (L2 norm) or not
        :param minimal_fisher_weight: float, the minimal value in fisher weights, used for tackling the potential numerical issues
        :return:
        """
        def get_param_squared_gradients(model: nn.Module, param_names_to_merge: list):
            """
            get the squared gradients of parameters
            :param model: nn.Module, model
            :param param_names_to_merge: list, list of parameter names that need to be merged
            :return:
            """
            param_squared_gradients = {param_name: param_value.grad.detach(
            ) ** 2 for param_name, param_value in model.named_parameters() if param_name in param_names_to_merge}
            return param_squared_gradients

        def get_models_fisher_norm(models_to_merge_param_dict: dict, models_to_merge_fisher_weights_list: list):
            """
            get normalization of fisher weights of all the models that need to be merged
            :param models_to_merge_param_dict: dict, dictionary of list, where key is the parameter name,
            value is a list of the corresponding parameters of all the models that need to be merged
            :param models_to_merge_fisher_weights_list: list, list of dictionaries with length len(models_to_merge),
            each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
            :return:
            """
            # dict, key is parameter name, value is a Tensor with shape (num_models_to_merge, )
            models_fisher_norm_dict = {}
            # compute L2 norm over models for each parameter
            for param_name, _ in models_to_merge_param_dict.items():
                # Tensor, shape (num_models_to_merge, *fisher_weight_shape)
                models_fisher = torch.stack([model_to_merge_fisher_weights[param_name]
                                            for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list], dim=0)
                dims = [dim_idx for dim_idx in range(1, models_fisher.dim())]
                # Tensor, shape (num_models_to_merge, ), compute L2 norm for each parameter
                models_fisher_norm = torch.norm(models_fisher, dim=dims)
                models_fisher_norm_dict[param_name] = models_fisher_norm

            # Tensor, shape (num_models_to_merge, num_parameters)
            models_fisher_norm = torch.stack(
                [models_fisher_norm for models_fisher_norm in models_fisher_norm_dict.values()], dim=1)
            # Tensor, shape (num_models_to_merge, ), compute L2 norm over all the parameters
            models_fisher_norm = torch.norm(models_fisher_norm, dim=1)
            return models_fisher_norm

        def merging_with_fisher_weights(models_to_merge_param_dict: dict, models_to_merge_fisher_weights_list: list, fisher_scaling_coefficients: torch.Tensor,
                                        normalize_fisher_weight: bool = True, minimal_fisher_weight: float = 1e-6):
            """
            merge parameters of different models with computed fisher weights
            :param models_to_merge_param_dict: dict, dictionary of list, where key is the parameter name,
            value is a list of the corresponding parameters of all the models that need to be merged
            :param models_to_merge_fisher_weights_list: list, list of dictionaries with length len(models_to_merge),
            each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
            :param fisher_scaling_coefficients: torch.Tensor, scaling coefficients to merge fisher weights
            :param normalize_fisher_weight: boolean, whether to normalize fisher weights (L2 norm) or not
            :param minimal_fisher_weight: float, the minimal value in fisher weights, used for tackling the potential numerical issues
            :return:
            """
            # dict, dictionary of model parameters
            merged_params = {}

            if normalize_fisher_weight:
                # Tensor, shape (num_models_to_merge, ), L2 norm over all the parameters of models that need to be merged
                models_fisher_norm = get_models_fisher_norm(models_to_merge_param_dict=models_to_merge_param_dict,
                                                            models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list)

            for param_name, param_value_list in models_to_merge_param_dict.items():
                # shape (num_models_to_merge, *parameter_shape)
                param_values = torch.stack(param_value_list, dim=0)
                # Tensor, shape (num_models_to_merge, *fisher_weight_shape), use minimal_fisher_weight to solve the potential numerical issues
                models_to_merge_fisher_weights = torch.stack([model_to_merge_fisher_weights[param_name]
                                                              for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list], dim=0) + minimal_fisher_weight

                # Tensor, shape (num_models_to_merge, 1, 1, ...)
                reshaped_scaling_coefficients = fisher_scaling_coefficients.reshape(
                    -1, *[1 for _ in range(param_values.dim() - 1)]).to(param_values.device)

                if normalize_fisher_weight:
                    # Tensor, shape (num_models_to_merge, )
                    _models_fisher_norm = 1.0 / \
                        (models_fisher_norm + minimal_fisher_weight)
                    normalized_models_fisher_norm = _models_fisher_norm / _models_fisher_norm.sum()
                    normalized_models_fisher_norm = normalized_models_fisher_norm.reshape(
                        -1, *[1 for _ in range(param_values.dim() - 1)])
                    reshaped_scaling_coefficients = reshaped_scaling_coefficients * \
                        normalized_models_fisher_norm

                # shape (*parameter_shape)
                numerator = (reshaped_scaling_coefficients *
                             models_to_merge_fisher_weights * param_values).sum(dim=0)

                # shape (*parameter_shape)
                denominator = (reshaped_scaling_coefficients *
                               models_to_merge_fisher_weights).sum(dim=0)

                merged_param = numerator / denominator
                merged_params[param_name] = merged_param
            return merged_params

        # dictionary of list, where key is the parameter name,
        # value is a list of the corresponding parameters of all the models that need to be merged
        models_to_merge_param_dict = defaultdict(list)

        # list of dictionaries with length len(models_to_merge),
        # each dictionary records the fisher weights (matrix or vector) of parameters for each model that needs to be merged
        models_to_merge_fisher_weights_list = []

        assert len(models_to_merge) == len(trainers) == len(
            nums_fisher_examples), "sizes of lists are not identical!"

        for model_idx, (model_to_merge, trainer, num_fisher_examples) in enumerate(zip(models_to_merge, trainers, nums_fisher_examples)):
            model_to_merge=model_to_merge.cuda()
            param_dict = {param_name: param_value for param_name,
                          param_value in model_to_merge.named_parameters()}
            # exclude parameter whose name matches element in exclude_param_names_regex
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(
                param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)

            for param_name in param_names_to_merge:
                models_to_merge_param_dict[param_name].append(
                    param_dict[param_name])

            # list of dictionaries with length (num_fisher_examples // batch_size) or (num_fisher_examples // batch_size) + 1,
            # each dictionary records the fisher weights of parameters for model_to_merge computed by examples in a batch
            batches_fisher_weights_list = []

            num_computed_examples = 0
            train_dataloader = trainer.get_train_dataloader()
            if num_fisher_examples % trainer._train_batch_size != 0:
                print(f"warning: the number of examples for computing fisher cannot be fully divided by the batch size for model {model_idx}, "
                      "which may lead to a slightly different number of the actually used examples.")
            for step, inputs in tqdm(enumerate(train_dataloader), desc=f"computing fisher weights for model {model_idx}"):
                if num_computed_examples >= num_fisher_examples:
                    break
                inputs = trainer._prepare_inputs(inputs)
                outputs = model_to_merge(**inputs)
                # Tensor, shape (batch_size, num_label_classes)
                logits = outputs.logits
                # compute fisher weights for regression task
                if logits.shape[-1] == 1:
                    # use the label information to compute loss and obtain gradients
                    mse_loss = outputs.loss
                    model_to_merge.zero_grad()
                    mse_loss.backward()
                    # dict, fisher weights of a batch
                    batch_fisher_weights = get_param_squared_gradients(
                        model=model_to_merge, param_names_to_merge=param_names_to_merge)
                # compute fisher weights for classification task
                else:
                    # use detach() to detach from the computation graph
                    # Tensor, shape (batch_size, num_label_classes)
                    labels_probabilities = torch.softmax(
                        logits, dim=-1).detach()
                    labels_log_probabilities = torch.log_softmax(
                        logits, dim=-1)
                    # sqrt labels_probabilities, since torch.sqrt(labels_probabilities) would be squared in the following squared gradients
                    labels_expectations = torch.sqrt(
                        labels_probabilities) * labels_log_probabilities
                    # sum over label classes and batch dimension
                    sum_labels_expectations = labels_expectations.sum(
                        dim=-1).sum(dim=0)
                    model_to_merge.zero_grad()
                    sum_labels_expectations.backward()
                    # dict, fisher weights of a batch
                    batch_fisher_weights = get_param_squared_gradients(
                        model=model_to_merge, param_names_to_merge=param_names_to_merge)

                batches_fisher_weights_list.append(batch_fisher_weights)
                num_computed_examples += trainer._train_batch_size
            del inputs,train_dataloader,logits
            torch.cuda.empty_cache()

            model_to_merge_fisher_weights = {}
            for batch_fisher_weights in batches_fisher_weights_list:
                for key in batch_fisher_weights:
                    if key not in model_to_merge_fisher_weights:
                        model_to_merge_fisher_weights[key] = batch_fisher_weights[key]
                    else:
                        model_to_merge_fisher_weights[key] += batch_fisher_weights[key]

            # mean over batches
            for key in model_to_merge_fisher_weights:
                model_to_merge_fisher_weights[key] /= num_computed_examples
            models_to_merge_fisher_weights_list.append(
                model_to_merge_fisher_weights)
            # models_to_merge_fisher_weights_list.append(
                # model_to_merge_fisher_weights.cpu())
            
            del model_to_merge,model_to_merge_fisher_weights
            torch.cuda.empty_cache()

        # for model_idx, (model_to_merge, trainer, num_fisher_examples) in enumerate(zip(models_to_merge, trainers, nums_fisher_examples)):
        #     model_to_merge = model_to_merge.cuda()
        #     param_dict = {param_name: param_value for param_name, param_value in model_to_merge.named_parameters()}
        #     param_names_to_merge = get_param_names_to_merge(list(param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
        #     del param_dict  # ✅ 立刻释放
        #     torch.cuda.empty_cache()

        #     model_to_merge_fisher_weights = {name: torch.zeros_like(model_to_merge.get_parameter(name)).cpu() for name in param_names_to_merge}

        #     num_computed_examples = 0
        #     train_dataloader = trainer.get_train_dataloader()

        #     if num_fisher_examples % trainer._train_batch_size != 0:
        #         print(f"warning: fisher examples not divisible by batch size for model {model_idx}")

        #     for step, inputs in tqdm(enumerate(train_dataloader), desc=f"computing fisher weights for model {model_idx}"):
        #         if num_computed_examples >= num_fisher_examples:
        #             break
        #         inputs = trainer._prepare_inputs(inputs)
        #         outputs = model_to_merge(**inputs)
        #         logits = outputs.logits

        #         if logits.shape[-1] == 1:  # regression
        #             mse_loss = outputs.loss
        #             model_to_merge.zero_grad()
        #             mse_loss.backward()
        #         else:  # classification
        #             labels_probabilities = torch.softmax(logits, dim=-1).detach()
        #             labels_log_probabilities = torch.log_softmax(logits, dim=-1)
        #             labels_expectations = torch.sqrt(labels_probabilities) * labels_log_probabilities
        #             sum_labels_expectations = labels_expectations.sum(dim=-1).sum(dim=0)
        #             model_to_merge.zero_grad()
        #             sum_labels_expectations.backward()

        #         # ✅ 直接累加，不用存 batch_fisher_weights_list
        #         batch_fisher_weights = get_param_squared_gradients(model=model_to_merge, param_names_to_merge=param_names_to_merge)
        #         batch_fisher_weights = {k: v.cpu() for k, v in get_param_squared_gradients(model=model_to_merge, param_names_to_merge=param_names_to_merge).items()}
        #         for key in batch_fisher_weights:
        #             model_to_merge_fisher_weights[key] += batch_fisher_weights[key]

        #         num_computed_examples += trainer._train_batch_size

        #         # ✅ 清理中间变量
        #         del outputs, logits
        #         if "mse_loss" in locals(): del mse_loss
        #         if "labels_probabilities" in locals(): del labels_probabilities
        #         if "labels_log_probabilities" in locals(): del labels_log_probabilities
        #         if "labels_expectations" in locals(): del labels_expectations
        #         if "sum_labels_expectations" in locals(): del sum_labels_expectations
        #         torch.cuda.empty_cache()

        #     # ✅ 取平均
        #     for key in model_to_merge_fisher_weights:
        #         model_to_merge_fisher_weights[key] /= num_computed_examples

        #     # ✅ 移到 CPU，避免显存增长
        #     models_to_merge_fisher_weights_list.append({k: v.cpu() for k, v in model_to_merge_fisher_weights.items()})

        #     # ✅ 删除当前模型
        #     del model_to_merge, model_to_merge_fisher_weights, train_dataloader, trainer
        #     torch.cuda.empty_cache()

        # merging with fisher weights
        # if fisher_scaling_coefficients is None, then set the fisher weights of different models to contribute equally
        if fisher_scaling_coefficients is None:
            fisher_scaling_coefficients = torch.ones(
                len(models_to_merge)) / len(models_to_merge)
        else:
            assert isinstance(fisher_scaling_coefficients,
                              list), "wrong type of fisher_scaling_coefficients, should be list!"
            assert len(fisher_scaling_coefficients) == len(
                models_to_merge), "mismatched length of fisher_scaling_coefficients!"
            fisher_scaling_coefficients = torch.Tensor(
                fisher_scaling_coefficients)
        
        # models_to_merge_param_dict = {k: v.cpu() for k, v in models_to_merge_param_dict.items()}
        models_to_merge_param_dict = {k: [p.cpu() for p in v] for k, v in models_to_merge_param_dict.items()}
        models_to_merge_fisher_weights_list = [{k: v.cpu() for k, v in model_fisher_dict.items()} for model_fisher_dict in models_to_merge_fisher_weights_list]
        # print(models_to_merge_fisher_weights_list)
        # merging with fisher weights
        merged_params = merging_with_fisher_weights(models_to_merge_param_dict=models_to_merge_param_dict, models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list,
                                                    fisher_scaling_coefficients=fisher_scaling_coefficients, normalize_fisher_weight=normalize_fisher_weight, minimal_fisher_weight=minimal_fisher_weight)

        return merged_params

    def regmean_merging(self, models_to_merge: list, trainers: list, exclude_param_names_regex: list, nums_regmean_examples: list, reduce_non_diagonal_ratio: float = 1.0):
        """
        regmean merging method
        :param models_to_merge: list, individual models that need to be merged
        :param trainers: list, trainers of individual models
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param nums_regmean_examples: list, numbers of examples to compute regmean weights
        :param reduce_non_diagonal_ratio: float, reduce non-diagonal elements in regmean weights by multiplying this scalar
        :return:
        """
        # def compute_regmean_weights(module_name: str):
        #     """
        #     compute the regmean weights, a hook function to deal with each module's input
        #     :param module_name: str, module name
        #     :return:
        #     """
        #     def hook(module: nn.Module, input: tuple, output: torch.Tensor):
        #         # Tensor, shape (batch_size, sequence_length, hidden_dim)
        #         x = input[0].detach()
        #         batch_num_actual_examples = x.shape[0]
        #         # Tensor, shape (batch_size * sequence_length, hidden_dim)
        #         x = x.reshape(-1, x.shape[-1])
        #         # Tensor, shape (hidden_dim, hidden_dim)
        #         xtx = torch.matmul(x.transpose(0, 1), x)
        #         # store the averaged weights in regmean_weights
        #         if module_name not in regmean_weights.keys():
        #             regmean_weights[module_name] = xtx / x.shape[0]
        #             num_computed_examples[module_name] = x.shape[0]
        #             num_actual_examples[module_name] = batch_num_actual_examples
        #         else:
        #             regmean_weights[module_name] = (regmean_weights[module_name] * num_computed_examples[module_name] + xtx) / (
        #                 num_computed_examples[module_name] + x.shape[0])
        #             num_computed_examples[module_name] += x.shape[0]
        #             num_actual_examples[module_name] += batch_num_actual_examples
        #     return hook
        def compute_regmean_weights(module_name: str):
            def hook(module: nn.Module, input: tuple, output: torch.Tensor):
                # x is likely on same device as model (cuda)
                x = input[0].detach()
                batch_num_actual_examples = x.shape[0]
                x_flat = x.reshape(-1, x.shape[-1])               # (B*L, H)
                xtx = torch.matmul(x_flat.transpose(0, 1), x_flat)  # (H, H) on model device (likely cuda)

                # Move the small-ish xtx to CPU immediately to avoid accumulating GPU memory.
                xtx_cpu = xtx.cpu()

                if module_name not in regmean_weights.keys():
                    regmean_weights[module_name] = xtx_cpu / x_flat.shape[0]
                    num_computed_examples[module_name] = x_flat.shape[0]
                    num_actual_examples[module_name] = batch_num_actual_examples
                else:
                    # accumulate on CPU (numerically stable enough)
                    prev = regmean_weights[module_name]
                    prev_count = num_computed_examples[module_name]
                    regmean_weights[module_name] = (prev * prev_count + xtx_cpu) / (prev_count + x_flat.shape[0])
                    num_computed_examples[module_name] += x_flat.shape[0]
                    num_actual_examples[module_name] += batch_num_actual_examples

                # clean up local GPU tensors asap
                del x, x_flat, xtx
                torch.cuda.empty_cache()
            return hook

        def reduce_non_diagonal_elements(regmean_weights: torch.Tensor, reduce_non_diagonal_ratio: float):
            """
            reduce the non-diagonal elements in regmean_weights
            :param regmean_weights: Tensor, shape (hidden_dim, hidden_dim), input regmean weights
            :param reduce_non_diagonal_ratio: float, reduce non-diagonal elements in regmean weights by multiplying this scalar
            :return:
            """
            # diagonal matrix with (1 - reduce_non_diagonal_ratio) as elements
            diag_weights = torch.diag(torch.ones(
                regmean_weights.shape[0]) - reduce_non_diagonal_ratio).to(regmean_weights.device)
            # matrix with reduce_non_diagonal_ratio as elements
            non_diag_weights = torch.zeros_like(
                diag_weights).fill_(reduce_non_diagonal_ratio)
            # diagonal elements are unchanged, while non-diagonal elements are multiplied by reduce_non_diagonal_ratio
            return regmean_weights * (diag_weights + non_diag_weights)

        def merging_with_regmean_weights(models_to_merge_param_dict: dict, models_to_merge_regmean_weights_list: list, reduce_non_diagonal_ratio: float = 1.0):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            merged_params = {}

            for param_name, param_value_list in models_to_merge_param_dict.items():
                if param_name.endswith(".weight"):
                    module_name = param_name[:-len(".weight")]
                    if module_name in models_to_merge_regmean_weights_list[0].keys():
                        # move per-model regmean mats (CPU) to device and compute
                        module_regmean_weights_list = []
                        param_multiplied_results = []
                        for model_idx, cpu_regmean_dict in enumerate(models_to_merge_regmean_weights_list):
                            module_regmean_weights = cpu_regmean_dict[module_name].to(device)   # move to GPU
                            module_regmean_weights = reduce_non_diagonal_elements(regmean_weights=module_regmean_weights, reduce_non_diagonal_ratio=reduce_non_diagonal_ratio)
                            module_regmean_weights_list.append(module_regmean_weights)
                            # param_value_list are CPU tensors -> move to device, transpose
                            param = param_value_list[model_idx].to(device)
                            param_multiplied_results.append(torch.matmul(module_regmean_weights, param.transpose(0,1)))

                        sum_module_regmean_weights = sum(module_regmean_weights_list)
                        sum_param_multiplied_results = sum(param_multiplied_results)

                        # inv_sum_module_regmean_weights = torch.inverse(sum_module_regmean_weights)
                        # inv_sum_module_regmean_weights = torch.linalg.pinv(sum_module_regmean_weights)
                        # try:
                        #     inv_sum_module_regmean_weights = torch.inverse(sum_module_regmean_weights)
                        # except RuntimeError as e:
                        #     if 'singular' in str(e):
                        #         print(f"Matrix is singular, using pseudo-inverse for module {module_name}")
                        #         inv_sum_module_regmean_weights = torch.linalg.pinv(sum_module_regmean_weights)
                        #     else:
                        #         raise e
                        # merged_param = torch.matmul(inv_sum_module_regmean_weights, sum_param_multiplied_results).transpose(0,1)
                        try:
                            inv_sum_module_regmean_weights = torch.inverse(sum_module_regmean_weights)
                        except RuntimeError as e:
                            if "singular" in str(e):
                                print(f" -> Pseudo-inverse also failed, falling back to simple parameter averaging")
                                inv_sum_module_regmean_weights = None  # trigger averaging
                            else:
                                raise e

                        if inv_sum_module_regmean_weights is not None:
                            merged_param = torch.matmul(inv_sum_module_regmean_weights, sum_param_multiplied_results).transpose(0, 1)
                        else:
                            # fallback: simple average of params
                            merged_param = torch.mean(torch.stack(param_value_list, dim=0), dim=0)

                        # move merged param back to CPU for storage (optional)
                        merged_params[param_name] = merged_param.cpu().clone()

                        # free GPU copies immediately
                        for t in module_regmean_weights_list:
                            del t
                        for t in param_multiplied_results:
                            del t
                        del sum_module_regmean_weights, sum_param_multiplied_results, inv_sum_module_regmean_weights, merged_param
                        torch.cuda.empty_cache()
                        continue
                # fallback: average on CPU
                merged_params[param_name] = torch.stack(param_value_list, dim=0).mean(dim=0)
            return merged_params
        # def merging_with_regmean_weights(models_to_merge_param_dict: dict, models_to_merge_regmean_weights_list: list, reduce_non_diagonal_ratio: float = 1.0):
        #     """
        #     merge parameters of different models with computed regmean weights
        #     :param models_to_merge_param_dict: dict, dictionary of list, where key is the parameter name,
        #     value is a list of the corresponding parameters of all the models that need to be merged
        #     :param models_to_merge_regmean_weights_list: list, list of dictionaries with length len(models_to_merge),
        #     each dictionary records the regmean weights (matrix) of parameters for each model that needs to be merged, key is module name
        #     :param reduce_non_diagonal_ratio: float, reduce non-diagonal elements in regmean weights by multiplying this scalar
        #     :return:
        #     """
        #     # dict, dictionary of model parameters
        #     merged_params = {}

        #     for param_name, param_value_list in models_to_merge_param_dict.items():
        #         merged_by_regmean = False
        #         # only perform regmean merging on the "weight" parameter of Linear module
        #         if param_name.endswith(".weight"):
        #             module_name = param_name[:-len(".weight")]
        #             if module_name in models_to_merge_regmean_weights_list[0].keys():
            #                 # two lists with length num_models_to_merge
        #                 param_multiplied_results, module_regmean_weights_list = [], []
        #                 for model_idx, model_to_merge_regmean_weights in enumerate(models_to_merge_regmean_weights_list):
        #                     # Tensor, shape (hidden_dim, hidden_dim)
        #                     module_regmean_weights = model_to_merge_regmean_weights[module_name]

        #                     # reduce non-diagonal elements
        #                     module_regmean_weights = reduce_non_diagonal_elements(
        #                         regmean_weights=module_regmean_weights, reduce_non_diagonal_ratio=reduce_non_diagonal_ratio)
        #                     module_regmean_weights_list.append(
        #                         module_regmean_weights)

        #                     model_to_merge_param = param_value_list[model_idx]
        #                     # since the weight shape of Linear module is (output_size, input_size), we need to transpose it
        #                     param_multiplied_results.append(torch.matmul(
        #                         module_regmean_weights, model_to_merge_param.transpose(0, 1)))

        #                 # sum up module_regmean_weights and param_multiplied_results over all individual models
        #                 sum_module_regmean_weights = sum(
        #                     module_regmean_weights_list)
        #                 sum_param_multiplied_results = sum(
        #                     param_multiplied_results)

        #                 # get the inverse matrix
        #                 inv_sum_module_regmean_weights = torch.inverse(
        #                     sum_module_regmean_weights)
        #                 # merge parameters with regmean
        #                 merged_param = torch.matmul(
        #                     inv_sum_module_regmean_weights, sum_param_multiplied_results)
        #                 # transpose to the original shape of "weight" in Linear module
        #                 merged_params[param_name] = merged_param.transpose(
        #                     0, 1)
        #                 merged_by_regmean = True
        #         # use average merging for parameters whose names are not end with ".weight" or not in Linear module
        #         if not merged_by_regmean:
        #             merged_params[param_name] = torch.stack(
        #                 param_value_list, dim=0).mean(dim=0)

        #     return merged_params

        # dictionary of list, where key is the parameter name,
        # value is a list of the corresponding parameters of all the models that need to be merged
        models_to_merge_param_dict = defaultdict(list)

        # list of dictionaries with length len(models_to_merge),
        # each dictionary records the regmean weights (matrix) of parameters for each model that needs to be merged
        models_to_merge_regmean_weights_list = []

        # iterate each individual model that needs to be merged
        with torch.no_grad():
            for model_idx, (model_to_merge, trainer, num_regmean_examples) in enumerate(zip(models_to_merge, trainers, nums_regmean_examples)):
                model_to_merge=model_to_merge.cuda()
                param_dict = {param_name: param_value for param_name,
                              param_value in model_to_merge.named_parameters()}
                # exclude parameter whose name matches element in exclude_param_names_regex
                param_names_to_merge = get_param_names_to_merge(input_param_names=list(
                    param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)

                for param_name in param_names_to_merge:
                    models_to_merge_param_dict[param_name].append(
                        param_dict[param_name])

                linear_modules_to_merge = get_modules_to_merge(
                    model=model_to_merge, include_module_types=[nn.Linear])
                handles = []
                # dictionary, regmean matrices for each linear module inputs
                regmean_weights = {}
                # dictionary, number of examples (multiplied the sequence length) used for computing regmean matrices
                num_computed_examples = {}
                # dictionary, number of actual examples used for computing regmean matrices
                num_actual_examples = {}

                for module_name, linear_module_to_merge in linear_modules_to_merge.items():
                    # register a hook in the forward process
                    handle = linear_module_to_merge.register_forward_hook(
                        compute_regmean_weights(module_name=module_name))
                    handles.append(handle)
                    del handle
                    torch.cuda.empty_cache()

                train_dataloader = trainer.get_train_dataloader()
                if num_regmean_examples % trainer._train_batch_size != 0:
                    print(f"warning: the number of examples for computing regmean cannot be fully divided by the batch size for model {model_idx}, "
                          "which may lead to a slightly different number of the actually used examples.")
                for step, inputs in tqdm(enumerate(train_dataloader), desc=f"computing regmean weights for model {model_idx}"):
                    if len(num_actual_examples) > 0 and list(num_actual_examples.values())[0] >= num_regmean_examples:
                        break
                    inputs = trainer._prepare_inputs(inputs)
                    outputs = model_to_merge(**inputs)
                     # 立即清理，避免显存堆积
                    del outputs
                    del inputs
                    torch.cuda.empty_cache()




                models_to_merge_regmean_weights_list.append(regmean_weights)

                # remove the added hook
                for handle in handles:
                    handle.remove()
    
                del regmean_weights
                del num_computed_examples
                del num_actual_examples
                del linear_modules_to_merge
                del model_to_merge
                torch.cuda.empty_cache()
            # merging with regmean weights
            merged_params = merging_with_regmean_weights(models_to_merge_param_dict=models_to_merge_param_dict, models_to_merge_regmean_weights_list=models_to_merge_regmean_weights_list,
                                                         reduce_non_diagonal_ratio=reduce_non_diagonal_ratio)

        return merged_params

    def ties_merging(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, param_value_mask_rate: float = 0.8, scaling_coefficient: float = 1.0):
        """
        ties merging method
        :param merged_model: nn.Module, the merged model
        :param models_to_merge: list, individual models that need to be merged
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
        :param scaling_coefficient: float, scaling coefficient to merge the task vectors
        :return:
        """
        def task_vector_param_dict_to_single_vector(task_vector: TaskVector):
            """
            convert parameter dictionary in task vector to a single vector
            :param task_vector: TaskVector, task vector
            :return:
            """
            task_vector_param_dict = copy.deepcopy(
                task_vector.task_vector_param_dict)
            sorted_task_vector_param_dict = OrderedDict(
                sorted(task_vector_param_dict.items()))

            # Tensor, shape (num_total_params, )
            return nn.utils.parameters_to_vector([param.flatten() for param in sorted_task_vector_param_dict.values()])

        def single_vector_to_task_vector_param_dict(single_vector: torch.Tensor, task_vector: TaskVector):
            """
            convert a single vector to parameter dictionary in task vector
            :param single_vector: Tensor, single vector that contain all parameters in task_vector.task_vector_param_dict
            :param task_vector: TaskVector, task vector
            :return:
            """
            task_vector_param_dict = copy.deepcopy(
                task_vector.task_vector_param_dict)
            sorted_task_vector_param_dict = OrderedDict(
                sorted(task_vector_param_dict.items()))

            nn.utils.vector_to_parameters(
                single_vector, sorted_task_vector_param_dict.values())

            return sorted_task_vector_param_dict

        def mask_smallest_magnitude_param_values(flattened_models_to_merge_param: torch.Tensor, param_value_mask_rate: float = 0.8):
            """
            mask the smallest-magnitude parameter values (set to zeros) based on parameter value mask rate
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
            :param param_value_mask_rate: float, mask rate of the smallest-magnitude parameter values
            :return:
            """
            # num_models_to_merge, num_total_params = flattened_models_to_merge_param.shape
            num_mask_params = int(
                flattened_models_to_merge_param.shape[1] * param_value_mask_rate)

            # Tensor, shape (num_models_to_merge, 1), find the num_mask_params-th smallest magnitude element of all the parameters in each individual model
            kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(
                k=num_mask_params, dim=1, keepdim=True)
            # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
            mask = flattened_models_to_merge_param.abs() >= kth_values

            return flattened_models_to_merge_param * mask

        def get_param_signs(flattened_models_to_merge_param: torch.Tensor):
            """
            get the signs for each parameter in flattened_models_to_merge_param, computed over individual models that need to be merged
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
            :return:
            """
            # Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
            param_signs = torch.sign(
                flattened_models_to_merge_param.sum(dim=0))
            # Tensor, shape (, ), a scalar, replace 0 in param_signs to the major sign in param_signs
            majority_sign = torch.sign(param_signs.sum(dim=0))
            param_signs[param_signs == 0] = majority_sign
            return param_signs

        def disjoint_merge(flattened_models_to_merge_param: torch.Tensor, param_signs: torch.Tensor):
            """
            disjoint merge that only keeps the parameter values in individual models whose signs are the same as the param_signs, and calculates the averaged parameters.
            :param flattened_models_to_merge_param: Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
            :param param_signs: Tensor, shape (num_total_params, ), the signs of parameters aggregated across individual models that need to be merged
            :return:
            """
            # Tensor, shape (num_models_to_merge, num_total_params), where True is for parameters that we want to preserve
            param_to_preserve_mask = ((param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)) | (
                (param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
            # Tensor, shape (num_models_to_merge, num_total_params), the preserved parameters
            param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask

            # Tensor, shape (num_total_params, ), the number of models whose parameters can be preserved
            num_models_param_preserved = (
                param_to_preserve != 0).sum(dim=0).float()
            # Tensor, shape (num_total_params, ), the averaged flattened parameters
            merged_flattened_param = torch.sum(
                param_to_preserve, dim=0) / torch.clamp(num_models_param_preserved, min=1.0)

            return merged_flattened_param

        assert isinstance(
            scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

        models_to_merge_task_vectors = [TaskVector(pretrained_model=merged_model, finetuned_model=model_to_merge,
                                                   exclude_param_names_regex=exclude_param_names_regex) for model_to_merge in models_to_merge]

        flattened_models_to_merge_param = [task_vector_param_dict_to_single_vector(
            task_vector=task_vector) for task_vector in models_to_merge_task_vectors]
        # Tensor, shape (num_models_to_merge, num_total_params), flattened parameters of individual models that need to be merged
        flattened_models_to_merge_param = torch.vstack(
            flattened_models_to_merge_param)

        with torch.no_grad():
            # Tensor, shape (num_models_to_merge, num_total_params), mask the smallest-magnitude parameter values using param_value_mask_rate
            flattened_models_to_merge_param = mask_smallest_magnitude_param_values(
                flattened_models_to_merge_param=flattened_models_to_merge_param, param_value_mask_rate=param_value_mask_rate)

            # Tensor, shape (num_total_params, ), get the signs for each parameter in flattened_models_to_merge_param
            param_signs = get_param_signs(
                flattened_models_to_merge_param=flattened_models_to_merge_param) 
            # Tensor, shape (num_total_params, ), disjoint merge
            # merged_flattened_param = disjoint_merge(
            #     flattened_models_to_merge_param=flattened_models_to_merge_param, param_signs=param_signs)
            # 假设 flattened_models_to_merge_param 是一个 List[Tensor]，每个 tensor 是某个模型的某层参数
            flattened_models_to_merge_param = flattened_models_to_merge_param.cpu()
            param_signs = param_signs.cpu()
            # 在 CPU 上执行合并操作
            merged_flattened_param = disjoint_merge(
                flattened_models_to_merge_param=flattened_models_to_merge_param,
                param_signs=param_signs
            )

            # 合并后的结果如果还需要参与 GPU 上训练或微调，再放回 CUDA：
            merged_flattened_param = merged_flattened_param.to("cuda")

            # merged parameter dictionary
            merged_task_vector_param_dict = single_vector_to_task_vector_param_dict(
                single_vector=merged_flattened_param, task_vector=models_to_merge_task_vectors[0])
            merged_task_vector = TaskVector(
                task_vector_param_dict=merged_task_vector_param_dict)
            # combine with parameters of the merged model based on scaling coefficient
            merged_params = merged_task_vector.combine_with_pretrained_model(
                pretrained_model=merged_model, scaling_coefficient=scaling_coefficient)

        return merged_params


    def get_merged_model(self, merged_model: nn.Module, models_to_merge: list, exclude_param_names_regex: list, trainers: list = None, scaling_coefficient: float = 1.0,
                         nums_fisher_examples: list = None, fisher_scaling_coefficients: list = None, normalize_fisher_weight: bool = True, minimal_fisher_weight: float = 1e-6,
                         nums_regmean_examples: list = None, reduce_non_diagonal_ratio: float = 1.0, param_value_mask_rate: float = 0.8, lr=0.01, epochs=50, granularity="elementwise"):

        # merged_params, dict of parameters
        pre_merged_model = copy.deepcopy(merged_model)
        if self.merging_method_name == "average_merging":
            merged_params = self.average_merging(
                models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex)
        elif self.merging_method_name == "task_arithmetic":
            merged_params = self.task_arithmetic(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                                 scaling_coefficient=scaling_coefficient)
        elif self.merging_method_name == "adamerging":
            merged_params = self.adamerging(merged_model=merged_model, models_to_merge=models_to_merge, trainers=trainers,
                                            exclude_param_names_regex=exclude_param_names_regex, lr=lr, epochs=epochs, granularity=granularity)
        elif self.merging_method_name == "fisher_merging":
            merged_params = self.fisher_merging(models_to_merge=models_to_merge, trainers=trainers, exclude_param_names_regex=exclude_param_names_regex,
                                                nums_fisher_examples=nums_fisher_examples, fisher_scaling_coefficients=fisher_scaling_coefficients,
                                                normalize_fisher_weight=normalize_fisher_weight, minimal_fisher_weight=minimal_fisher_weight)
        elif self.merging_method_name == "regmean_merging":
            merged_params = self.regmean_merging(models_to_merge=models_to_merge, trainers=trainers, exclude_param_names_regex=exclude_param_names_regex,
                                                 nums_regmean_examples=nums_regmean_examples, reduce_non_diagonal_ratio=reduce_non_diagonal_ratio)
        elif self.merging_method_name == "ties_merging":
            merged_params = self.ties_merging(merged_model=merged_model, models_to_merge=models_to_merge, exclude_param_names_regex=exclude_param_names_regex,
                                              param_value_mask_rate=param_value_mask_rate, scaling_coefficient=scaling_coefficient)
        else:
            raise NotImplementedError(
                f"unsupported for merging_method_name {self.merging_method_name}!")
        
        self.copy_params_to_model(params=merged_params, model=pre_merged_model)

        return pre_merged_model

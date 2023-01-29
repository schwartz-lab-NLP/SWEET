import torch
from torch.nn import CrossEntropyLoss, MSELoss




#
# class ConstLambdaLoss():
#
#     def __init__(self, loss_lambda: float = 0.0) -> None:
#         self.loss_lambda = float(loss_lambda)
#         self.cross_entropy = CrossEntropyLoss(reduce=False)
#
#     def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         # prev_logits = logits[:-1,...]
#         # if prev_logits.shape[0] > 0:
#         #     prev_predictions = torch.argmax(prev_logits, dim = 2)
#         #     has_been_learnd = (prev_predictions == target.repeat(prev_logits.shape[0], 1)).type(torch.float)
#         # else:
#         #     has_been_learnd = torch.zeros((1, prev_logits.shape[1], 1), device= logits.device)
#         # has_been_learnd.detach()
#         # lambda_term = (1 - self.loss_lambda * torch.mean(has_been_learnd, dim=0))
#         # return torch.mean(self.cross_entropy(logits[-1,...], target) * lambda_term)
#         loss_list = []
#         has_been_learned = torch.zeros((logits.shape[0], target.shape[0]), device=logits.device)
#         for i in range(logits.shape[0]):
#             current_logits = logits[i,...]
#             loss = self.cross_entropy(current_logits, target)
#             if i > 0:
#                 loss = loss * (1 - self.loss_lambda * torch.mean(has_been_learned[0:i, ...], axis=0))
#             loss = torch.mean(loss)
#             loss_list.append(loss)
#             predictions = torch.argmax(current_logits, axis=1)
#             has_been_learned[i] = (predictions == target).detach()
#         return torch.sum(torch.stack(loss_list, dim=0))
#
#
#
# class ConstLambdaLossOnes():
#
#     def __init__(self, loss_lambda: float = 0.0) -> None:
#         self.loss_lambda = float(loss_lambda)
#         self.cross_entropy = CrossEntropyLoss(reduce=False)
#
#     def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         loss_list = []
#         has_been_learned = torch.ones((logits.shape[0], target.shape[0]), device=logits.device)
#         for i in range(logits.shape[0]):
#             current_logits = logits[i,...]
#             loss = self.cross_entropy(current_logits, target)
#             if i > 0:
#                 loss = loss * (1 - self.loss_lambda * torch.mean(has_been_learned[0:i, ...], axis=0))
#             loss = torch.mean(loss)
#             loss_list.append(loss)
#         return torch.sum(torch.stack(loss_list, dim=0))
#
#
# class DynamicLambdaLoss():
#
#     def __init__(self, lambda_func: callable = lambda x:x) -> None:
#         self.lambda_func = lambda_func
#         self.cross_entropy = CrossEntropyLoss(reduce=False)
#
#     def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         loss_list = []
#         has_been_learned = torch.zeros((logits.shape[0], target.shape[0]), device=logits.device)
#         for i in range(logits.shape[0]):
#             current_logits = logits[i,...]
#             loss = self.cross_entropy(current_logits, target)
#             if i > 0:
#                 loss = loss * (1 - self.loss_lambda * torch.mean(has_been_learned[0:i, ...], axis=0))
#             loss = torch.mean(loss)
#             loss_list.append(loss)
#             predictions = torch.argmax(current_logits, axis=1)
#             has_been_learned[i] = (predictions == target).detach()
#         return torch.sum(torch.stack(loss_list, dim=0))
#
#
# class ConstLambdaLossWithOr():
#
#     def __init__(self, loss_lambda: float = 0.0) -> None:
#         self.loss_lambda = float(loss_lambda)
#         self.cross_entropy = CrossEntropyLoss(reduce=False)
#
#     def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         loss_list = []
#         has_been_learned = torch.zeros((logits.shape[0], target.shape[0]), device=logits.device)
#         for i in range(logits.shape[0]):
#             current_logits = logits[i,...]
#             loss = self.cross_entropy(current_logits, target)
#             if i > 0:
#                 loss = loss * (1 - self.loss_lambda * torch.max(has_been_learned[0:i, ...], axis=0).values)
#             loss = torch.mean(loss)
#             loss_list.append(loss)
#             predictions = torch.argmax(current_logits, axis=1)
#             has_been_learned[i] = (predictions == target).detach()
#         return torch.sum(torch.stack(loss_list, dim=0))
#
#
# class ConstLambdaLossPrevLayer():
#
#     def __init__(self, loss_lambda: float = 0.0) -> None:
#         self.const_lambda_loss = ConstLambdaLoss(loss_lambda)
#
#     def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return self.const_lambda_loss(logits[-2, ...])


class MultiExitCrossEntropyLoss():
    def __init__(self):
        self.cross_entropy = CrossEntropyLoss(reduce=True)

    def __call__(self, logits: torch.Tensor, target: torch.Tensor, lte_logits: torch.Tensor = None, gold_classifier = None) -> torch.Tensor:
        loss_list = []

        for i in range(logits.shape[0]):
            loss = torch.mean(self.cross_entropy(logits[i,...], target))
            loss_list.append(loss)

        if gold_classifier is not None:
            return torch.Tensor(loss_list[gold_classifier])
        return torch.sum(torch.stack(loss_list, dim=0))




class MultiExitMSELossForLTE():
    def __init__(self):
        self.lte_loss_fct = MSELoss()

    def __call__(self, logits: torch.Tensor, target: torch.Tensor, lte_logits: torch.Tensor = None, gold_classifier = None) -> torch.Tensor:
        lte_loss_list = []

        for i in range(logits.shape[0]):
            lte_gold = torch.eq(torch.argmax(logits[i, ...], dim=1), target).float().unsqueeze(1)  # 0 for wrong/continue, 1 for right/exit
            loss = torch.mean(self.lte_loss_fct(lte_logits[i, ...], lte_gold))
            lte_loss_list.append(loss)

        if gold_classifier is not None:
            return torch.Tensor(lte_loss_list[gold_classifier])
        return torch.sum(torch.stack(lte_loss_list, dim=0))




import torch
from torch.autograd.gradcheck import zero_gradients


def compute_jacobian(inputs, output):
    """
    NOTE: prefer `torch.autograd.functional.jacobian` from Pytorch >= 1.5.0
    :param inputs: (batch, input_features)
    :param output: (batch, output_features)
    :return: jacobian: (batch, output_features, input_features)
    """
    assert inputs.requires_grad, "Inputs must require gradients."

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)


def get_logdet(jacobian):
    """
    Use QR factorisation to compute log absolute determinant of the jacobian matrix.
    NOTE: prefer `torch.slogdet` from Pytorch >= 0.4
    :param jacobian: (M, M)
    :return: log-determinant jacobian
    """
    Q, R = torch.qr(jacobian)
    log_det = torch.log(torch.diag(R).abs()).sum()
    return log_det

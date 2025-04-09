# import torch
# import numpy as np
# import torch.nn as nn
# from torch.autograd import Variable


# def _concat(xs):
#   return torch.cat([x.view(-1) for x in xs])


# class Architect(object):

#   def __init__(self, model, args):
#     self.network_momentum = args.momentum
#     self.network_weight_decay = args.weight_decay
#     self.model = model
#     self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
#         lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

#   def _compute_unrolled_model(self, input, target, eta, network_optimizer):
#     loss = self.model._loss(input, target)
#     theta = _concat(self.model.parameters()).data
#     try:
#       moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
#     except:
#       moment = torch.zeros_like(theta)
#     dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
#     unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
#     return unrolled_model

#   def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
#     self.optimizer.zero_grad()
#     if unrolled:
#         self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
#     else:
#         self._backward_step(input_valid, target_valid)
#     self.optimizer.step()

#   def _backward_step(self, input_valid, target_valid):
#     loss = self.model._loss(input_valid, target_valid)
#     loss.backward()

#   def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
#     unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
#     unrolled_loss = unrolled_model._loss(input_valid, target_valid)

#     unrolled_loss.backward()
#     dalpha = [v.grad for v in unrolled_model.arch_parameters()]
#     vector = [v.grad.data for v in unrolled_model.parameters()]
#     implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

#     for g, ig in zip(dalpha, implicit_grads):
#       g.data.sub_(eta, ig.data)

#     for v, g in zip(self.model.arch_parameters(), dalpha):
#       if v.grad is None:
#         v.grad = Variable(g.data)
#       else:
#         v.grad.data.copy_(g.data)

#   def _construct_model_from_theta(self, theta):
#     model_new = self.model.new()
#     model_dict = self.model.state_dict()

#     params, offset = {}, 0
#     for k, v in self.model.named_parameters():
#       v_length = np.prod(v.size())
#       params[k] = theta[offset: offset+v_length].view(v.size())
#       offset += v_length

#     assert offset == len(theta)
#     model_dict.update(params)
#     model_new.load_state_dict(model_dict)
#     return model_new.cuda()

#   def _hessian_vector_product(self, vector, input, target, r=1e-2):
#     R = r / _concat(vector).norm()
#     for p, v in zip(self.model.parameters(), vector):
#       p.data.add_(R, v)
#     loss = self.model._loss(input, target)
#     grads_p = torch.autograd.grad(loss, self.model.arch_parameters(), allow_unused=True)

#     for p, v in zip(self.model.parameters(), vector):
#       p.data.sub_(2*R, v)
#     loss = self.model._loss(input, target)
#     grads_n = torch.autograd.grad(loss, self.model.arch_parameters(), allow_unused=True)

#     for p, v in zip(self.model.parameters(), vector):
#       p.data.add_(R, v)

#     return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]



import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect(object):
    """
    This architect module implements the differentiable architecture search updates.
    It computes unrolled gradients and the implicit Hessian-vector product needed to update
    the architecture parameters, for use with your transformer model.
    """
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=args.arch_learning_rate, betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay
        )

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        """
        Compute the unrolled model by taking a single gradient descent step on the training data.
        This function is model-agnostic; it calls `self.model._loss` and uses the model parameters.
        """
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        
        # Try extracting momentum from the optimizer if available.
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters())
            moment = moment.mul_(self.network_momentum)
        except Exception:
            moment = torch.zeros_like(theta)
            
        # Compute gradient of loss with respect to model parameters and add weight decay term.
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        
        # Create the updated theta using the learning rate eta.
        unrolled_theta = theta.sub(eta, moment + dtheta)
        unrolled_model = self._construct_model_from_theta(unrolled_theta)
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        """
        Update the architecture parameters. If unrolled is True, compute the unrolled gradients.
        """
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(
                input_train, target_train, input_valid, target_valid, eta, network_optimizer
            )
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        """
        Standard backward step using the validation loss.
        """
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    # def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    #     """
    #     Perform a backward step with unrolled optimization.
    #     """
    #     unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    #     unrolled_loss = unrolled_model._loss(input_valid, target_valid)
        
    #     # Compute gradients on the unrolled model.
    #     unrolled_loss.backward()
    #     dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        
    #     # Compute gradients with respect to the model parameters.
    #     vector = [v.grad.data for v in unrolled_model.parameters()]
    #     implicit_grads = self._hessian_vector_product(vector, input_train, target_train)
        
    #     # Adjust the architecture gradients with the implicit gradients.
    #     for g, ig in zip(dalpha, implicit_grads):
    #         g.data.sub_(eta, ig.data)
        
    #     # Copy the computed gradients back to the original architecture parameters.
    #     for v, g in zip(self.model.arch_parameters(), dalpha):
    #         if v.grad is None:
    #             v.grad = Variable(g.data)
    #         else:
    #             v.grad.data.copy_(g.data)

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
      unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
      unrolled_loss = unrolled_model._loss(input_valid, target_valid)
      
      unrolled_loss.backward()
      # Get gradients (may include None's)
      dalpha = [v.grad for v in unrolled_model.arch_parameters()]
      vector = [v.grad.data for v in unrolled_model.parameters()]
      implicit_grads = self._hessian_vector_product(vector, input_train, target_train)
      
      # Replace None gradients with zeros and update:
      new_dalpha = []
      for param, g, ig in zip(unrolled_model.arch_parameters(), dalpha, implicit_grads):
          if g is None:
              g = torch.zeros_like(param)
          if ig is None:
              ig = torch.zeros_like(param)
          # Update gradient: g = g - eta * ig
          new_dalpha.append(g - eta * ig)
      
      # Copy the updated gradients back to the original architecture parameters.
      for v, g in zip(self.model.arch_parameters(), new_dalpha):
          if v.grad is None:
              v.grad = g.clone()  # Using clone() to ensure independence.
          else:
              v.grad.data.copy_(g.data)


    def _construct_model_from_theta(self, theta):
        """
        Construct a new model with parameters given by theta.
        The transformer model should implement a .new() method that returns a clone.
        """
        model_new = self.model.new()
        model_dict = self.model.state_dict()
        params, offset = {}, 0
        
        # Update parameters from the theta vector.
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        # Perturb the parameters in the positive direction
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters(), allow_unused=True)
        
        # Perturb the parameters in the negative direction (from the original)
        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters(), allow_unused=True)
        
        # Restore the original parameters
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        
        # Replace None with zeros and compute the finite-difference Hessian-vector product
        hessian_vector = []
        for param, gp, gn in zip(self.model.arch_parameters(), grads_p, grads_n):
            gp = gp if gp is not None else torch.zeros_like(param)
            gn = gn if gn is not None else torch.zeros_like(param)
            hessian_vector.append((gp - gn).div_(2 * R))
        
        return hessian_vector
#     return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
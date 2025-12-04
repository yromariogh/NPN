import torch





class DeepNorm(torch.nn.Module):

    def __init__(self):
        super(DeepNorm, self).__init__()
    
    def forward(self, g, x, y, H=None):

        return torch.norm(g(x, y, H),dim=1)
    
    def grad(self, g, x, y, H=None):
        x = x.requires_grad_()
        norm = torch.norm(g(x, y, H),1)

        return torch.autograd.grad(norm, x, create_graph=True, grad_outputs=torch.ones_like(norm))[0]
    

class LearnedL2(torch.nn.Module):
    def __init__(self, G):

        super(LearnedL2, self).__init__()
        self.G = G

    def forward(self, y, x, H=None):
        r = self.G(x) - H(x)
            
        r = r.reshape(r.shape[0],-1)
        return 1/2*torch.norm(r,p=2,dim=1)**2
    
    def grad(self, G, x, H=None):
        x = x.requires_grad_()
        norm = self.forward(G, x, H)
 
        return torch.autograd.grad(norm, x, create_graph=True, grad_outputs=torch.ones_like(norm))[0]



class WeightedL2(torch.nn.Module):
    r"""
        L2 fidelity

        .. math::
           f(\mathbf{x}) =  \frac{1}{2}||\forwardLinear(\mathbf{x}) - \mathbf{y}||^2_2

    """
    def __init__(self):
        super(WeightedL2, self).__init__()

    def forward(self, x, y, H=None, M=None):
        r""" Computes the L2 fidelity term.

        Args:
            x (torch.Tensor): The image to be reconstructed.
            y (torch.Tensor): The measurement data to be reconstructed.
            H (function, optional): The forward model. Defaults to None.

        Returns:
            torch.Tensor: The L2 fidelity term.
        """
        r = M(H(x) - y)
        r = r.view(r.shape[0],-1)
        return 1/2*torch.norm(r,p=2,dim=1)**2

    def grad(self, x, y, H=None, M = None, transform=None):
        r'''
        Compute the gradient of the L2 fidelity term.

        .. math::
            \nabla f(\mathbf{x}) = \nabla \frac{1}{2}||\forwardLinear(\mathbf{x}) - \mathbf{y}||^2_2

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): The measurement data to be reconstructed.
            H (function): Forward model.   

        Returns:
            torch.Tensor: Gradient of the L1 fidelity term. 
        '''
        x = x.requires_grad_()
        norm = self.forward(x,y,H,M)
 
        return torch.autograd.grad(norm, x, create_graph=True, grad_outputs=torch.ones_like(norm))[0]
 

class L2(torch.nn.Module):
    r"""
        L2 fidelity

        .. math::
           f(\mathbf{x}) =  \frac{1}{2}||\forwardLinear(\mathbf{x}) - \mathbf{y}||^2_2

    """
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, x, y, H=None):
        r""" Computes the L2 fidelity term.

        Args:
            x (torch.Tensor): The image to be reconstructed.
            y (torch.Tensor): The measurement data to be reconstructed.
            H (function, optional): The forward model. Defaults to None.

        Returns:
            torch.Tensor: The L2 fidelity term.
        """
        r = H(x) - y
        r = r.reshape(r.shape[0],-1)
        return 1/2*torch.norm(r,p=2,dim=1)**2
        # return 1/2*torch.norm( H(x) - y,p=2,)**2
    
    def grad(self, x, y, H=None, transform=None):
        r'''
        Compute the gradient of the L2 fidelity term.

        .. math::
            \nabla f(\mathbf{x}) = \nabla \frac{1}{2}||\forwardLinear(\mathbf{x}) - \mathbf{y}||^2_2

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): The measurement data to be reconstructed.
            H (function): Forward model.   

        Returns:
            torch.Tensor: Gradient of the L1 fidelity term. 
        '''
        x = x.requires_grad_()
        norm = self.forward(x,y,H)
 
        return torch.autograd.grad(norm, x, create_graph=True, grad_outputs=torch.ones_like(norm))[0]

import torch

class Huber(torch.nn.Module):
    r"""
        Huber fidelity

        .. math::
           f(\mathbf{x}) = \sum_i 
           \begin{cases} 
           \frac{1}{2} (H(\mathbf{x}) - \mathbf{y})_i^2 & \text{if } |(H(\mathbf{x}) - \mathbf{y})_i| \leq \delta \\
           \delta \left( |(H(\mathbf{x}) - \mathbf{y})_i| - \frac{1}{2} \delta \right) & \text{otherwise}
           \end{cases}
    """
    def __init__(self, delta=1.0):
        super(Huber, self).__init__()
        self.delta = delta

    def forward(self, x, y, H=None):
        r""" Computes the Huber fidelity term.

        Args:
            x (torch.Tensor): The image to be reconstructed.
            y (torch.Tensor): The measurement data.
            H (function, optional): The forward model.

        Returns:
            torch.Tensor: The Huber fidelity term (batch-wise).
        """
        r = H(x) - y
        r = r.view(r.shape[0], -1)
        abs_r = torch.abs(r)
        quadratic = 0.5 * (r ** 2)
        linear = self.delta * (abs_r - 0.5 * self.delta)
        loss = torch.where(abs_r <= self.delta, quadratic, linear)
        return loss.sum(dim=1)  # Sum over features, returns shape (batch,)

    def grad(self, x, y, H=None, transform=None):
        r'''
        Compute the gradient of the Huber fidelity term.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): The measurement data.
            H (function): Forward model.

        Returns:
            torch.Tensor: Gradient of the Huber fidelity term.
        '''
        x = x.requires_grad_()
        norm = self.forward(x, y, H)
        return torch.autograd.grad(norm, x, create_graph=True, grad_outputs=torch.ones_like(norm))[0]


class L1(torch.nn.Module):
    r"""
        L1 fidelity

        .. math::
            f(\mathbf{x}) = ||\forwardLinear(\mathbf{x}) - \mathbf{y}||_1
    """
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, x, y, H):
        r""" Computes the L1 fidelity term.

        Args:
            x (torch.Tensor): The image to be reconstructed.
            y (torch.Tensor): The measurement data to be reconstructed.
            H (function): The forward model.

        Returns:
            torch.Tensor: The L1 fidelity term.
        """
        
        return torch.norm( H(x) - y,p=1)
    
    def grad(self, x, y, H):
        r'''
        Compute the gradient of the L1 fidelity term.

        .. math::
            \nabla f(\mathbf{x}) = \nabla \frac{1}{2}||\forwardLinear(\mathbf{x}) - \mathbf{y}||_1
            

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): The measurement data to be reconstructed.
            H (function): Forward model.   

        Returns:
            torch.Tensor: Gradient of the L1 fidelity term. 
        '''
        x = x.requires_grad_()

        return torch.autograd.grad(self.forward(x,y, H), x, create_graph=True)[0]
    


        



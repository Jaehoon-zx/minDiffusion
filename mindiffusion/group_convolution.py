import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class C4Group(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.register_buffer('identity', torch.Tensor([0.]))
        self.order = torch.tensor(4)

    def elements(self):
        """ 
        out: a tensor containing all group elements in this group.
        """

        out = torch.linspace(
            start=0,
            end=2 * np.pi * float(self.order - 1) / float(self.order),
            steps=self.order,
            device=self.identity.device
        )

        return out

    def product(self, h1, h2):
        """ 
        h1: group element 1 
        h2: group element 2
        out: group product of two group elements
        """
        
        out = torch.remainder(h1 + h2, 2 * np.pi)

        return out

    def inverse(self, h):
        """ 
        h: group element
        out: group inverse of the group element 
        """

        out = torch.remainder(-h, 2 * np.pi)

        return out

    def matrix_representation(self, h):
        """ 
        h: group element
        out: matrix representation in R^2 for the group element.
        """
        cos_t = torch.cos(h)
        sin_t = torch.sin(h)

        out = torch.tensor([
            [cos_t, -sin_t],
            [sin_t, cos_t]
        ], device=self.identity.device)

        return out

    def left_action_on_R2(self, batch_h, batch_x):
        """
        batch_h: batch of group elements (b)
        batch_x: vectors defined in R2   (i, x, y)
        out: left action of the elements on a set of vectors in R2 (b, x, y, i)
        """

        batch_h_matrix = torch.stack([self.matrix_representation(h) for h in batch_h]) # matrix representation
        out = torch.einsum('boi,ixy->bxyo', batch_h_matrix, batch_x) # left action
        out = out.roll(shifts=1, dims=-1) # swap x and y coordinates
        return out

    def left_action_on_H(self, batch_h, batch_h_prime):
        """ 
        batch_h: batch of group elements (b)
        batch_h_prime: batch of group elements (b)
        out: batchwise left group actions (b, b)
        """
        transformed_batch_h = self.product(batch_h.repeat(batch_h_prime.shape[0], 1),
                                           batch_h_prime.unsqueeze(-1))
        return transformed_batch_h

    # this is not the problem
    def normalize_group_elements(self, h):
        """ Normalize values of group elements to range between -1 and 1.
        The group elements range from 0 to 2pi * (self.order - 1) / self.order,
        so we normalize by

        @param h: A group element.
        :return:
        """
        largest_elem = 2 * np.pi * (self.order - 1) / self.order

        return (2*h / largest_elem) - 1.


class Z2P4Kernel(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.group = C4Group()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # create spatial kernel grid. These are the coordinates on which our kernel weights are defined.
        self.register_buffer("grid_R2", torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, self.kernel_size),
            torch.linspace(-1, 1, self.kernel_size),
        )).to(self.group.identity.device))

        # transform the grid by the elements in this group.
        self.register_buffer("transformed_grid_R2", self.create_transformed_grid_R2())

        # create and initialize a set of weights
        self.weight = torch.nn.Parameter(torch.zeros((
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        ), device=self.group.identity.device))

        # Initialize weights using kaiming uniform intialisation
        torch.nn.init.kaiming_uniform_(self.weight.data, a=np.sqrt(5))

    def create_transformed_grid_R2(self):
        """
        Transform the created grid by the group action of each group element.
        This yields a grid (over H) of spatial grids (over R2). In other words,
        a list of grids, each index of which is the original spatial grid transformed by
        a corresponding group element in H.
        """
        # Obtain all group elements.
        group_elements = self.group.elements()

        # Transform the grid defined over R2 with the sampled group elements.
        transformed_grid = self.group.left_action_on_R2(
            self.group.inverse(group_elements),
            self.grid_R2
        )
        return transformed_grid

    def sample(self):
        """ 
        Sample convolution kernels for a given number of group elements
        out: filter bank extending over all input channels,
            containing kernels transformed for all output group elements.
        """
        # We fold the output channel dim into the input channel dim; this allows
        # us to use the torch grid_sample function.
        weight = self.weight.view(
            1,
            self.out_channels * self.in_channels,
            self.kernel_size,
            self.kernel_size
        )

        # We want a transformed set of weights for each group element so
        # we repeat the set of spatial weights along the output group axis
        weight = weight.repeat(self.group.elements().numel(), 1, 1, 1)

        # Sample the transformed kernels
        transformed_weight = torch.nn.functional.grid_sample(
            weight,
            self.transformed_grid_R2,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        # Separate input and output channels
        transformed_weight = transformed_weight.view(
            self.group.elements().numel(),
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        )

        # Put the output channel dimension before the output group dimension.
        transformed_weight = transformed_weight.transpose(0, 1)

        return transformed_weight


class Z2P4Conv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.kernel = Z2P4Kernel(in_channels, out_channels, kernel_size=kernel_size)
        self.padding = padding

    def forward(self, x):
        """ 
        Perform Z2-P4 convolution
        x: [batch_dim, in_channels, spatial_dim_1, spatial_dim_2]
        out: [batch_dim, out_channels, num_group_elements, spatial_dim_1, spatial_dim_2]
        """

        # obtain convolution kernels transformed under the group
        conv_kernels = self.kernel.sample()

        # apply convolution
        x = torch.nn.functional.conv2d(
            input=x,
            weight=conv_kernels.reshape(
                self.kernel.out_channels * self.kernel.group.elements().numel(),
                self.kernel.in_channels,
                self.kernel.kernel_size,
                self.kernel.kernel_size
            ),
            padding = self.padding,
        ) # [batch_dim, out_channels * num_group_elements, spatial_dim_1,spatial_dim_2]
        
        # reshape
        x = x.view(
            -1,
            self.kernel.out_channels,
            self.kernel.group.elements().numel(),
            x.shape[-1],
            x.shape[-2]
        ) # [batch_dim, out_channels, num_group_elements, spatial_dim_1, spatial_dim_2]

        return x

class P4P4Kernel(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.group = C4Group()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create a spatial kernel grid
        self.register_buffer("grid_R2", torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, self.kernel_size),
            torch.linspace(-1, 1, self.kernel_size),
        )).to(self.group.identity.device))

        # The kernel grid now also extends over the group H, as our input
        # feature maps contain an additional group dimension
        self.register_buffer("grid_H", self.group.elements())
        self.register_buffer("transformed_grid_R2xH", self.create_transformed_grid_R2xH())

        # create and initialise a set of weights, we will interpolate these
        # to create our transformed spatial kernels. Note that our weight
        # now also extends over the group H
        self.weight = torch.nn.Parameter(torch.zeros((
            self.out_channels,
            self.in_channels,
            self.group.elements().numel(), # this is different from the lifting convolution
            self.kernel_size,
            self.kernel_size
        ), device=self.group.identity.device))

        # initialize weights using kaiming uniform intialisation
        torch.nn.init.kaiming_uniform_(self.weight.data, a=np.sqrt(5))

    def create_transformed_grid_R2xH(self):
        """

        """
        # Sample the group
        group_elements = self.group.elements()

        # Transform the grid defined over R2 with the sampled group elements
        transformed_grid_R2 = self.group.left_action_on_R2(
            self.group.inverse(group_elements),
            self.grid_R2
        )

        # Transform the grid defined over H with the sampled group elements
        transformed_grid_H = self.group.left_action_on_H(
            self.group.inverse(group_elements), self.grid_H
        )

        # Rescale values to between -1 and 1, we do this to please the torch grid_sample
        # function.
        transformed_grid_H = self.group.normalize_group_elements(transformed_grid_H)

        # Create a combined grid as the product of the grids over R2 and H
        # repeat R2 along the group dimension, and repeat H along the spatial dimension
        # to create a [output_group_elem, num_group_elements, kernel_size, kernel_size, 3] grid
        transformed_grid = torch.cat(
            (
                transformed_grid_R2.view(
                    group_elements.numel(),
                    1,
                    self.kernel_size,
                    self.kernel_size,
                    2,
                ).repeat(1, group_elements.numel(), 1, 1, 1),
                transformed_grid_H.view(
                    group_elements.numel(),
                    group_elements.numel(),
                    1,
                    1,
                    1,
                ).repeat(1, 1, self.kernel_size, self.kernel_size, 1, )
            ),
            dim=-1
        )
        return transformed_grid

    def sample(self):
        """ Sample convolution kernels for a given number of group elements

        should return:
        :return kernels: filter bank extending over all input channels,
            containing kernels transformed for all output group elements.
        """

        # fold the output channel dim into the input channel dim; this allows
        # us to use the torch grid_sample function
        weight = self.weight.view(
            1,
            self.out_channels * self.in_channels,
            self.group.elements().numel(),
            self.kernel_size,
            self.kernel_size
        )

        # we want a transformed set of weights for each group element so
        weight = weight.repeat(self.group.elements().numel(), 1, 1, 1, 1)

        # sample the transformed kernels,
        transformed_weight = torch.nn.functional.grid_sample(
            weight,
            self.transformed_grid_R2xH,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        # Separate input and output channels. Note we now have a notion of
        # input and output group dimensions in our weight matrix!
        transformed_weight = transformed_weight.view(
            self.group.elements().numel(), # Output group elements (like in the lifting convolutoin)
            self.out_channels,
            self.in_channels,
            self.group.elements().numel(), # Input group elements (due to the additional dimension of our feature map)
            self.kernel_size,
            self.kernel_size
        )

        # Put the output channel dimension before the output group dimension.
        transformed_weight = transformed_weight.transpose(0, 1)

        return transformed_weight

class P4P4Conv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.kernel = P4P4Kernel(in_channels, out_channels, kernel_size=kernel_size)
        self.padding = padding

    def forward(self, x):
        """ 
        Perform P4-P4 convolution
        x: [batch_dim, in_channels, num_group_elements, spatial_dim_1, spatial_dim_2]
        out: [batch_dim, out_channels, num_group_elements, spatial_dim_1, spatial_dim_2]
        """

        # We now fold the group dimensions of our input into the input channel
        # dimension
        x = x.reshape(
            -1,
            x.shape[1] * x.shape[2],
            x.shape[3],
            x.shape[4]
        )

        # We obtain convolution kernels transformed under the group
        conv_kernels = self.kernel.sample()

        # apply convolution
        x = torch.nn.functional.conv2d(
            input=x,
            weight=conv_kernels.reshape(
                self.kernel.out_channels * self.kernel.group.elements().numel(),
                self.kernel.in_channels * self.kernel.group.elements().numel(),
                self.kernel.kernel_size,
                self.kernel.kernel_size
            ),
            padding = self.padding
        ) # [batch_dim, out_channels * num_group_elements, spatial_dim_1, spatial_dim_2]

        # reshape
        x = x.view(
            -1,
            self.kernel.out_channels,
            self.kernel.group.elements().numel(),
            x.shape[-1],
            x.shape[-2],
        ) # [batch_dim, out_channels, num_group_elements, spatial_dim_1, spatial_dim_2]

        return x


class P4Z2Pooling(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.mean(x, dim=2) #Average out the num_group_elements part
        # -> [batch_dim, out_channels, spatial_dim_1, spatial_dim_2]
        # x = x.squeeze()

        return x
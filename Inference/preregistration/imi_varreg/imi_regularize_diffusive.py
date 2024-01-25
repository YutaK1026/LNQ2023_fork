import torch
import torch.nn.functional as F
# DBEUG
from preregistration.imi_varreg.imi_debug_tools import debug_print


class ImiRegularizeDiffusive(torch.nn.Module):
    def __init__(self, img_size, alpha, tau=1.0, spacing=(1, 1, 1), use_spacing=True):
        super().__init__()
        self.dim = len(img_size) - 2
        self.spacing = spacing
        assert self.dim == 2 or self.dim == 3
        assert len(spacing) == self.dim
        self.Nx = img_size[-1]
        self.Ny = img_size[-2]
        self.Nz = img_size[-3] if self.dim == 3 else None
        self.alpha = alpha
        self.tau = tau
        weights = [self.alpha*self.tau] * self.dim  # Todo: check use of image-dimension in ITK code
        if use_spacing:
            mean_squared_spacing = sum([s*s for s in self.spacing]) / self.dim
            weights = [w * mean_squared_spacing / (s*s) for w, s in zip(weights, self.spacing)]
        self.solver_x = TriDiagonalLaplaceSolver(self.Nx, tau=weights[0])
        self.solver_y = TriDiagonalLaplaceSolver(self.Ny, tau=weights[1])
        self.solver_z = TriDiagonalLaplaceSolver(self.Nz, tau=weights[2]) if self.Nz is not None else None
        self.solve_semi_implicit = self.solve_semi_implicit_2d if self.Nz is None else self.solve_semi_implicit_3d

    def solve_semi_implicit_2d(self, field):
        assert field.shape[-1] == self.Nx
        assert field.shape[-2] == self.Ny
        assert len(field.shape) == 4  # N, C, H, W
        out_field = field.clone()  # NOT in-place  -- TODO: check if necessary
        # solve for x-direction
        out_field = self.solver_x(out_field)
        # solve for y-direction -- we have to permute because last dimension is solved
        out_field = self.solver_y(out_field.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return out_field

    def solve_semi_implicit_3d(self, field):
        assert field.shape[-1] == self.Nx
        assert field.shape[-2] == self.Ny
        assert field.shape[-3] == self.Nz
        assert len(field.shape) == 5  # N, C, D, H, W
        out_field = field.clone()  # NOT in-place  -- TODO: check if necessary
        # solve for x-direction
        out_field = self.solver_x(out_field)
        # solve for y-direction -- we have to permute because last dimension is solved
        out_field = self.solver_y(out_field.permute(0, 1, 2, 4, 3)).permute(0, 1, 2, 4, 3)
        # solve for z-direction -- we have to permute because last dimension is solved
        out_field = self.solver_z(out_field.permute(0, 1, 4, 3, 2)).permute(0, 1, 4, 3, 2)
        return out_field

    def forward(self, field):
        return self.solve_semi_implicit(field)

    def compute_loss(self, field):
        return diffusive_loss(field, spacing=self.spacing).mean()


class TriDiagonalLaplaceSolver(torch.nn.Module):
    def __init__(self, N, tau, dtype=torch.double):
        super().__init__()
        self.N = N
        debug_print(5, f"solver: N={N}, tau={tau}, a={1 - tau * -2}, b,c={0 - tau * 1}")
        # setup tri-diagonal matrix
        b = torch.ones(N - 1, dtype=dtype) * (0 - tau * 1)
        a = torch.ones(N, dtype=dtype) * (1 - tau * -2)
        c = torch.ones(N - 1, dtype=dtype) * (0 - tau * 1)
        c[0] = 0 - tau * 1
        b[-1] = 0 - tau * 1
        a[0] = (1 + tau)
        a[-1] = (1 + tau)
        # pre-compute
        for i in range(N-1):
            c[i] = c[i] / a[i]
            a[i+1] = a[i+1] - c[i] * b[i]
        self.register_buffer('a', a)
        self.register_buffer('b', b)
        self.register_buffer('c', c)

    def solve(self, x):
        # This function solves in-place !!!
        assert x.shape[-1] == self.N
        # forward pass
        for i in range(1, self.N):
            x[..., i] = x[..., i] - x[..., i-1] * self.c[i-1]
        # backward pass
        x[..., -1] = x[..., -1] / self.a[-1]
        for i in range(self.N-2, -1, -1):
            x[..., i] = (x[..., i] - x[..., i+1] * self.b[i]) / self.a[i]
        return x

    def forward(self, x):
        return self.solve(x)


#
#         FUNCTIONS
#
def diffusive_loss(displacement, spacing=(1, 1, 1)):
    if displacement.dim() == 4:
        return diffusive_loss_2d(displacement, spacing)
    else:
        return diffusive_loss_3d(displacement, spacing)


def diffusive_loss_2d(displacement, spacing=(1, 1)):
    """Compute the diffusive loss \|\nabla_x f\|^2 + \|\nabla_y f\|^2 of a given vector field.
    using finite differences.
    Args:
        displacement (DisplacementField): the field to compute the loss
    Returns:
        the computed loss per pixel and component
    """
    assert len(displacement.shape) == 4  # N,C,H,W layout
    assert displacement.shape[1] == 2
    # use central differences (same as ITK)
    dx = ((displacement[:, :, 1:-1, 2:] - displacement[:, :, 1:-1, :-2])/(2*spacing[0])).pow(2)  # divide by spacing for finite differences
    dy = ((displacement[:, :, 2:, 1:-1] - displacement[:, :, :-2, 1:-1])/(2*spacing[1])).pow(2)
    return F.pad(dx + dy, (1, 1, 1, 1))


def diffusive_loss_3d(displacement, spacing=(1, 1, 1)):
    """Compute the diffusive loss \|\nabla_x f\|^2 + \|\nabla_y f\|^2 of a given vector field.
    using finite differences.
    Args:
        displacement (DisplacementField): the field to compute the loss
    Returns:
        the computed loss per pixel and component
    """
    assert len(displacement.shape) == 5  # N,C,D,H,W layout
    assert displacement.shape[1] == 3  # field has 3d vectors
    # use central differences (same as ITK)
    dx = ((displacement[:, :, 1:-1, 1:-1, 2:] - displacement[:, :, 1:-1, 1:-1, :-2])/(2*spacing[0])).pow(2)   # missing: divide by spacing for finite differences
    dy = ((displacement[:, :, 1:-1, 2:, 1:-1] - displacement[:, :, 1:-1, :-2, 1:-1])/(2*spacing[1])).pow(2)
    dz = ((displacement[:, :, 2:, 1:-1, 1:-1] - displacement[:, :, :-2, 1:-1, 1:-1])/(2*spacing[2])).pow(2)
    return F.pad(dx + dy + dz, (1, 1, 1, 1, 1, 1))


if __name__ == '__main__':
    import argparse
    from imi_image_warp import load_image_sitk, load_field_sitk, sitk_image_to_tensor, tensor_image_to_sitk, write_image_sitk
    # parse the commandline
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', '-f', required=True, type=str, help='path for displacement or velocity field')
    parser.add_argument('--output', '-o', type=str, help='path for output image')
    parser.add_argument('--torch', action='store_true', help='Use pytorch functions for warping')
    parser.add_argument('--sitk', action='store_true', help='Use SimpleITK functions for warping')
    parser.add_argument('--space', type=str, default='world', help="Space of the field 'world'|'pixel'|'torch', default='world'")
    args = parser.parse_args()

    field = load_field_sitk(args.field)
    tensor_field, field_meta = sitk_image_to_tensor(field, return_meta_data=True)
    regularizer = ImiRegularizeDiffusive(tensor_field.size(), alpha=0.5, spacing=field_meta['spacing'])
    for i in range(5):
        loss = regularizer.compute_loss(tensor_field)
        tensor_field = regularizer(tensor_field)
        print('it:',i,'loss=',loss)
    if args.output:
        field = tensor_image_to_sitk(tensor_field, meta_data=field_meta)
        print('Save image ... ')
        write_image_sitk(field, args.output)
    print('Finished.')


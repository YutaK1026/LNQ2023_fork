import torch.nn.functional as F
import SimpleITK as sitk

#
#         FUNCTIONS
#
def compute_jacobian_det(displacement, spacing=(1, 1, 1)):
    if displacement.dim() == 4:
        return compute_jacobian_det_2d(displacement, spacing)
    else:
        return compute_jacobian_det_3d(displacement, spacing)


def _determinant_2x2(a00, a01, a10, a11):
    return a00 * a11 - a01 * a10  # compute determinant of 2x2 matrix


def compute_jacobian_det_2d(displacement, spacing=(1, 1)):
    assert displacement.dim() == 4  # assume 2D displacement field
    assert displacement.shape[1] == 2  # assume 2D displacement field in (N, 2, H, W)
    # compute for displacement  -- not transform
    # J(x+u(x)) = 1 + J(u(x))
    # use central differences (same as ITK)
    dx = (displacement[:, :, 1:-1, 2:] - displacement[:, :, 1:-1, :-2]) / (2 * spacing[0])  # divide by spacing for finite differences
    dy = (displacement[:, :, 2:, 1:-1] - displacement[:, :, :-2, 1:-1]) / (2 * spacing[1])
    J00, J01 = dx[:, 0, :, :], dx[:, 1, :, :]
    J10, J11 = dy[:, 0, :, :], dy[:, 1, :, :]
    Jdet = _determinant_2x2(J00, J01, J10, J11)  # compute determinant of 2x2 matrix
    # add value 1 for s.a.
    return F.pad(1.0 + Jdet, (1, 1, 1, 1), mode="constant", value=1.0).unsqueeze(1)  # change padding to (0,1,0,1) for forward differences


def compute_jacobian_det_3d(displacement, spacing=(1, 1, 1)):
    assert displacement.dim() == 5  # assume 3D displacement field
    assert displacement.shape[1] == 3  # assume 3D displacement field in (N, 3, D, H, W)
    # compute for displacement  -- not transform
    # J(x+u(x)) = I + J(u(x))
    # use central differences (same as ITK)
    dx = (displacement[:, :, 1:-1, 1:-1, 2:] - displacement[:, :, 1:-1, 1:-1, :-2]) / (2 * spacing[0])  # divide by spacing for finite differences
    dy = (displacement[:, :, 1:-1, 2:, 1:-1] - displacement[:, :, 1:-1, :-2, 1:-1]) / (2 * spacing[1])
    dz = (displacement[:, :, 2:, 1:-1, 1:-1] - displacement[:, :, :-2, 1:-1, 1:-1]) / (2 * spacing[2])
    # Add identity to jacobian
    J00, J01 , J02 = 1.0 + dx[:, 0, :, :, :], dx[:, 1, :, :, :], dx[:, 2, :, :, :]
    J10, J11 , J12 = dy[:, 0, :, :, :], 1.0 + dy[:, 1, :, :, :], dy[:, 2, :, :, :]
    J20, J21 , J22 = dz[:, 0, :, :, :], dz[:, 1, :, :, :], 1.0 + dz[:, 2, :, :, :]
    Jdet = J00 * _determinant_2x2(J11, J12, J21, J22) \
           - J01 * _determinant_2x2(J10, J12, J20, J22) \
           + J02 * _determinant_2x2(J10, J11, J20, J21)  # compute determinant of 3x3 matrix
    # Jdet = _determinant_3x3(J00, J01, J02, J10, J11, J12, J20, J21, J22)
    return F.pad(Jdet, (1, 1, 1, 1, 1, 1), mode="constant", value=1.0).unsqueeze(1)  # change padding to (0,1,0,1) for forward differences

#
#  Comparison / check functions
#
def jacobian_determinant_sitk(field_sitk):
    # fields need to have type sitk.sitkVectorFloat64
    field_sitk = sitk.Cast(field_sitk, sitk.sitkVectorFloat64)
    jacobian = sitk.DisplacementFieldJacobianDeterminant(field_sitk)
    return jacobian


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
    if args.output:
        if args.sitk:
            print('Compute jacobian (simpleITK)... ')
            jdet = jacobian_determinant_sitk(field)
        else:
            print('Compute jacobian (torch)... ')
            print(field_meta)
            print(tensor_field.shape)
            tensor_jdet = compute_jacobian_det(tensor_field, spacing=field_meta['spacing'])
            print(tensor_jdet.shape)
            jdet = tensor_image_to_sitk(tensor_jdet, meta_data=field_meta)
        print('Save image ... ')
        write_image_sitk(jdet, args.output)

    jaco_loss = JacobianRegulariserLoss(spacing=field_meta['spacing'])
    print(jaco_loss(tensor_field))
    neg_loss = NegativeJacobianLoss(spacing=field_meta['spacing'])
    print(neg_loss(tensor_field))
    print('Finished.')


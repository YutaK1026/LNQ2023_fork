import time
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from preregistration.imi_varreg.imi_regularized_warper import ImiRegularizedWarper
from preregistration.imi_varreg.imi_regularize_diffusive import diffusive_loss
from preregistration.imi_varreg.imi_jacobian import compute_jacobian_det
from preregistration.imi_varreg.imi_deformation import identity_displacement_field, warp_image_with_displacement_field, scaling_and_squaring
from preregistration.imi_varreg.imi_distance_ncc import NCC
from preregistration.imi_varreg.imi_image_pyramid import build_pyramid, pyrup
from preregistration.imi_varreg.imi_image_tools import crop_image, pad_image_to_size
from preregistration.imi_varreg.imi_debug_tools import debug_write_image_sitk, debug_print
from preregistration.imi_varreg.imi_stop_criterion import StopCriterion
from preregistration.imi_varreg.imi_image_sitk_tools import sitk_image_to_tensor, tensor_image_to_sitk
import preregistration.imi_varreg.imi_debug_tools



def imi_compute_statistics(fixed, moving, displ_field, spacing=(1, 1, 1)):
    distance_mse = F.mse_loss(fixed, moving)
    distance_l1 = F.l1_loss(fixed, moving)
    diff_loss = diffusive_loss(displ_field, spacing).mean()
    jacobian = compute_jacobian_det(displ_field, spacing)
    neg_jacobians = torch.count_nonzero((jacobian < 0))
    stat_values = {'mse': distance_mse, 'mae': distance_l1, 'reg_diff': diff_loss,
                   'log_jacobian': torch.log(torch.clamp(jacobian, 0.2, 5.0)).abs().mean(),
                   'std_jacobian': jacobian.std(), 'neg_jacobian': neg_jacobians}
    for key, values in stat_values.items():
        stat_values[key] = values.detach().cpu().item()
    return stat_values


def imi_variational_registration(fixed_image, moving_image, initial_field, spacing=(1, 1, 1),
                                 alpha=1.0, tau=0.5, iterations=10,
                                 loss_type='demons', regularizer='diffusive', diffeomorphic=False,
                                 optimizer_type='sgd', stop_policy='none'):
    # Setup loss function
    if loss_type == 'demons':
        loss_func = torch.nn.MSELoss(reduction='mean')
        gradient_scaling_type = 'demons'
    elif loss_type == 'mse':
        loss_func = torch.nn.MSELoss(reduction='mean')
        gradient_scaling_type = 'normalize'
    elif loss_type == 'ncc':
        ncc_loss = NCC()
        loss_func = ncc_loss.loss
        gradient_scaling_type = 'spacing'
    else:
        raise NotImplementedError(f"Unknown loss type '{loss_type}', supported losses: 'demons'|'mse'|'ncc'.")
    # Setup Warper and Regularizer
    warper_regularizer = ImiRegularizedWarper(fixed_image.size(), alpha=alpha, tau=tau, spacing=spacing,
                                              regularizer=regularizer, gradient_scaling=gradient_scaling_type,
                                              diffeomorphic=diffeomorphic, use_spacing=True).to(device=fixed_image.device)
    # Setup displacement/velocity field to optimize
    displacement_field = initial_field
    if displacement_field is None:
        displacement_field = identity_displacement_field(fixed_image.size(), device=fixed_image.device,
                                                         dtype=fixed_image.dtype)
    displacement_field.requires_grad_(True)
    # Setup optimizer
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD([displacement_field], lr=1)
    elif optimizer_type == 'nesterov':
        optimizer = torch.optim.SGD([displacement_field], lr=1, momentum=0.1, dampening=0.0, nesterov=True)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam([displacement_field], lr=0.01) #, betas=(0.1, 0.9999))
    else:
        raise NotImplementedError(f"Unknown optimizer type '{optimizer_type}', supported optimizers: 'sgd'|'nesterov'|'adam'.")

    stop_criterion = None
    if stop_policy == 'none':
        pass
    elif stop_policy == 'increase_count':
        stop_criterion = StopCriterion(use_increase_count=True, maximum_increase_count=10, line_regression_mode='none')
    elif stop_policy == 'line_fitting':
        stop_criterion = StopCriterion(use_increase_count=True, maximum_increase_count=10,
                                       line_regression_mode='standard', regression_line_slope_threshold=0.00000001)
    else:
        raise NotImplementedError(f"Unknown stop policy type '{stop_policy}', supported policies: 'none'|'increase_count'|'line_fitting'.")

    # START REGISTRATION
    best_loss, best_field, best_iter = 0, None, 0
    for iter in range(iterations):
        warped_image, regularizer_loss = warper_regularizer(moving_image, displacement_field)
        distance_loss = loss_func(fixed_image, warped_image)
        total_loss = distance_loss + alpha * regularizer_loss
        print(f"Iter {iter}: loss = {total_loss} (dist={distance_loss}, regul={regularizer_loss}).")
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if distance_loss < best_loss or iter == 0:
            best_loss, best_field, best_iter = distance_loss, displacement_field.detach().clone(), iter
        if stop_criterion is not None:
            stop_criterion.append(distance_loss)
            if stop_criterion.check_stop_criterion():
                break

    best_warped, _ = warper_regularizer(moving_image, best_field)
    print(f"Best loss {best_loss} at iteration {best_iter} -- {loss_func(fixed_image, best_warped)}.")
    return best_warped, best_field, best_loss


def normalize_image(image):
    min_val = image.min()
    max_val = image.max()
    factor = 1.0/(max_val - min_val)
    return (image - min_val) * factor, min_val, max_val


def denormalize_image(image, min_val, max_val):
    return image * (max_val - min_val) + min_val


def run_var_reg(sitk_fixed_image, sitk_moving_image, sitk_initial_field=None,
                levels: int = 3, iterations=100, alpha=1.0, tau=0.5,
                loss='demons', diffeo=True, normalize: bool = True,
                device='cpu', debug_level: int = 3):
    assert isinstance(sitk_fixed_image, sitk.Image)
    assert isinstance(sitk_moving_image, sitk.Image)
    assert isinstance(sitk_initial_field, sitk.Image) or sitk_initial_field is None

    # needed initialization
    preregistration.imi_varreg.imi_debug_tools.initialize_dbg_level(debug_level)

    fixed_image, fixed_meta = sitk_image_to_tensor(sitk_fixed_image, return_meta_data=True)
    moving_image, moving_meta = sitk_image_to_tensor(sitk_moving_image, return_meta_data=True)
    initial_field = sitk_image_to_tensor(sitk_initial_field, return_meta_data=False) \
        if sitk_initial_field is not None else None
    # check for image size
    if levels > 1:
        orig_image_size = fixed_image.shape[2:]
        possible_size = [(int(s / 2**(levels-1))) * 2**(levels-1) for s in orig_image_size]
        if orig_image_size != possible_size:
            # crop images
            print(f'Crop input images from size {orig_image_size} to size {possible_size}.')
            start_idx = (0, 0, 0) if len(possible_size) == 3 else (0, 0)
            fixed_image = crop_image(fixed_image, possible_size, start=start_idx)
            moving_image = crop_image(moving_image, possible_size, start=start_idx)
            initial_field = crop_image(initial_field, possible_size, start=start_idx) if initial_field is not None else None
            print(fixed_image.shape)

    fixed_image = fixed_image.to(device=device)
    moving_image = moving_image.to(device=device)
    initial_field = initial_field.to(device=device) if initial_field is not None else None

    start_time = time.time()

    image_spacing = fixed_meta['spacing']
    if normalize:
        print("Normalize images (distance before: {}).".format(F.mse_loss(fixed_image, moving_image)))
        fixed_image, fmin, fmax = normalize_image(fixed_image)
        moving_image, mmin, mmax = normalize_image(moving_image)

    fixed_pyramid = build_pyramid(fixed_image, max_level=levels)
    moving_pyramid = build_pyramid(moving_image, max_level=levels)
    init_field_pyramid = build_pyramid(initial_field, max_level=levels) if initial_field is not None else None
    level_init_field = init_field_pyramid[-1] if init_field_pyramid is not None else None
    for level in reversed(range(len(fixed_pyramid))):
        level_spacing = [s * 2**level for s in image_spacing]
        print(f"PROCESSING LEVEL {level} (spacing={level_spacing}, size={fixed_pyramid[level].shape}):")
        level_fixed_image = fixed_pyramid[level]
        level_moving_image = moving_pyramid[level]
        debug_write_image_sitk(5, level_fixed_image, f"pyreg_fixed_l{level}.nii.gz", fixed_meta)
        debug_write_image_sitk(5, level_moving_image, f"pyreg_moving_l{level}.nii.gz", fixed_meta)

        warped_image, field, best_loss = imi_variational_registration(level_fixed_image, level_moving_image,
                                                                      level_init_field, spacing=level_spacing,
                                                                      alpha=alpha, tau=tau,
                                                                      iterations=iterations, loss_type=loss,
                                                                      regularizer='diffusive', diffeomorphic=diffeo,
                                                                      optimizer_type='sgd',
                                                                      stop_policy='line_fitting' if level < 2 else 'none')
        # upsample computed field for next level
        level_init_field = pyrup(field.detach()) if level > 0 else None
        debug_write_image_sitk(5, warped_image.detach(), f"pyreg_warped_l{level}.nii.gz", fixed_meta)
        debug_write_image_sitk(5, field.detach(), f"pyreg_field_l{level}.nii.gz", fixed_meta)

    if normalize:
        print("Denormalize images.")
        fixed_image = denormalize_image(fixed_image, fmin, fmax)
        moving_image = denormalize_image(moving_image, mmin, mmax)
        warped_image = denormalize_image(warped_image, mmin, mmax)
        print("Denormalized image distance: {}".format(F.mse_loss(fixed_image, warped_image)))

    # check for image size
    if levels > 1:
        if orig_image_size != possible_size:
            # pad output images
            print(f'Pad output to size {orig_image_size}.')
            fixed_image = pad_image_to_size(fixed_image, orig_image_size, fill_mode='replicate')
            moving_image = pad_image_to_size(moving_image, orig_image_size, fill_mode='replicate')
            warped_image = pad_image_to_size(warped_image, orig_image_size, fill_mode='replicate')
            field = pad_image_to_size(field, orig_image_size, fill_mode='replicate')

    total_time = (time.time() - start_time)
    print(f"Registration finished in {total_time} seconds.")
    print(f"Best total loss: {best_loss}.")
    displacement_field = scaling_and_squaring(field, spacing=image_spacing) if diffeo else field
    warped_image = warp_image_with_displacement_field(moving_image, displacement_field, spacing=image_spacing)
    stats_dict = imi_compute_statistics(fixed_image, warped_image, displacement_field, image_spacing)
    debug_print(3, str(stats_dict))

    output_dict = {}
    output_dict['warped'] = tensor_image_to_sitk(warped_image, fixed_meta)
    output_dict['displacement'] = tensor_image_to_sitk(displacement_field, fixed_meta)
    output_dict['velocity'] = tensor_image_to_sitk(field, fixed_meta) if diffeo else None
    output_dict['best_loss'] = best_loss.item()
    output_dict.update(stats_dict)

    return output_dict


if __name__ == '__main__':
    import argparse
    import torch.nn.functional as F
    from imi_image_warp import load_image_sitk, load_field_sitk, sitk_image_to_tensor, tensor_image_to_sitk, write_image_sitk
    from imi_deformation import scaling_and_squaring
    from imi_regularize_diffusive import diffusive_loss
    from imi_jacobian import compute_jacobian_det

    # parse the commandline
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed', '-f', required=True, type=str, help='path for input fixed image')
    parser.add_argument('--moving', '-m', required=True, type=str, help='path for input moving image')
    parser.add_argument('--output', '-o', required=True, type=str, help='path for output warped image')
    parser.add_argument('--outdispl', '-d', type=str, help='path for output displacement field')
    parser.add_argument('--outvelo', '-v', type=str, help='path for output velocity field (if diffeomorphic)')
    parser.add_argument('--outcsv', type=str, help='path for output csv file with statistics')
    parser.add_argument('--initial_field', '-init', type=str, help='path for initial displacement or velocity field')
    parser.add_argument('--iterations', '-i', type=int, default=8, help='path for displacement or velocity field')
    parser.add_argument('--alpha', '-a', type=float, default=1.0, help='Regularization weight')
    parser.add_argument('--tau', '-t', type=float, default=0.5, help='Time step for displacement field update step.')
    parser.add_argument('--levels', '-l', type=int, default=3, help='path for displacement or velocity field')
    parser.add_argument('--diffeo', action='store_true', help='Use diffeomorphic registration')
    parser.add_argument('--normalize', action='store_true', help='Normalize images to 0-1')
    parser.add_argument('--loss', type=str, default='demons', help='Loss function to use.')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer type to use.')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--debug_level', '-x', type=int, default=3, help='Set debug level')
    args = parser.parse_args()

    fixed_image = load_image_sitk(args.fixed)
    moving_image = load_image_sitk(args.moving)
    initial_field = load_field_sitk(args.initial_field) if args.initial_field else None

    output_dict = run_var_reg(fixed_image, moving_image, initial_field, levels=args.level, iterations=args.iterations,
                              alpha=args.alpha, tau=args.tau, loss=args.loss, diffeo=args.diffeo,
                              normalize=args.normalize, device=args.device, debug_level=args.debug_level)

    print(f"Save output image to {args.output}.")
    write_image_sitk(output_dict['warped'], args.output)
    if args.outdispl:
        print(f"Save output displacement to {args.outdispl}.")
        write_image_sitk(output_dict['displacement'], args.outdispl)
    if args.outvelo:
        print(f"Save output velocity to {args.outvelo}.")
        assert args.diffeo
        write_image_sitk(output_dict['velocity'], args.outvelo)
    if args.outcsv:
        print(f"Save output to csv {args.outcsv}.")
        with open(args.outcsv, 'w') as f:
            args_attributes = ['iterations', 'levels', 'alpha', 'tau', 'diffeo', 'normalize', 'loss', 'optimizer']
            stats_keys = ['best_loss', 'mse', 'mae', 'reg_diff','log_jacobian','std_jacobian','neg_jacobian']
            f.write("# ")
            for header in args_attributes + stats_keys:
                f.write(f"{header};  ")
            f.write("\n")
            for attr in args_attributes:
                args_attr = getattr(args, attr)
                f.write(f"{args_attr};  ")
            for key in stats_keys:
                f.write(str(output_dict[key]) + "; ")
            f.write("\n")

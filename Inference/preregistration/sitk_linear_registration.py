import SimpleITK as sitk
import copy

# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    del metric_values
    del multires_iterations

# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations
    metric_values.append(registration_method.GetMetricValue())
    print(f"level {registration_method.GetCurrentLevel()}, it {registration_method.GetOptimizerIteration()} "
          f"lr {registration_method.GetOptimizerLearningRate()} it {len(metric_values)}: {metric_values[-1]}")

# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the
# metric_values list.
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))


def sitk_inverse_linear_transform(transform: sitk.Transform):
    inverse_transform = copy.deepcopy(transform)
    if not inverse_transform.SetInverse():
        raise ValueError(f"Can not compute inverse of {inverse_transform}")
    return inverse_transform


def sitk_transform_ct_image(transform, fixed_image, moving_image):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(transform)
    resample.SetDefaultPixelValue(-1024)
    transformed_image = resample.Execute(moving_image)
    return transformed_image

def sitk_transform_mask_image(transform, fixed_image, moving_image):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    resample.SetInterpolator(sitk.sitkLabelGaussian)
    resample.SetTransform(transform)
    resample.SetDefaultPixelValue(0)
    transformed_image = resample.Execute(moving_image)
    return transformed_image

def sitk_register_similarity(fixed_image, moving_image, init_translation=None):
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    initial_transform : sitk.Transform = sitk.CenteredTransformInitializer(fixed_image, moving_image,
                                                          sitk.Similarity3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    # print("Initial transform:", initial_transform, initial_transform.GetParameters())
    if init_translation is not None:
        params = list(initial_transform.GetParameters())
        print(f"init translation before: {params[3:6]}, translation after: {init_translation}")
        params[3:6] = init_translation  # works only for similarity transform
        initial_transform.SetParameters(params)
    # print(initial_transform)
    # print(fixed_image, moving_image)
    # transform_and_save_image(initial_transform, fixed_image, moving_image, "/tmp/init")

    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    registration_method.SetMetricAsCorrelation()
    registration_method.SetInterpolator(sitk.sitkLinear)
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10,
                                                      estimateLearningRate=sitk.ImageRegistrationMethod.EachIteration)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=True)

    # Connect all of the observers so that we can perform plotting during registration.
    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    final_transform = registration_method.Execute(fixed_image, moving_image)
    registration_method.GetOptimizerIteration()

    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    print('final_transform:', final_transform)

    return final_transform, moving_resampled


def sitk_register_affine(fixed_image, moving_image, initial_transform):
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    optimized_transform = sitk.AffineTransform(fixed_image.GetDimension())

    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    registration_method.SetMetricAsCorrelation()
    registration_method.SetInterpolator(sitk.sitkLinear)
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10,
                                                      estimateLearningRate=sitk.ImageRegistrationMethod.EachIteration,
                                                      maximumStepSizeInPhysicalUnits=0.5)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetMovingInitialTransform(initial_transform)
    registration_method.SetInitialTransform(optimized_transform)

    # Connect all of the observers so that we can perform plotting during registration.
    registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    final_transform = registration_method.Execute(fixed_image, moving_image)
    # Need to compose the transformations after registration.
    final_transform_v11 = sitk.CompositeTransform([initial_transform, optimized_transform])
    print('Final metric value: {0}'.format(final_transform))
    print('Final metric value: {0}'.format(final_transform_v11))

    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    # moving_resampled_test = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    # moving_resampled_test = sitk.Resample(moving_resampled_test, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    # sitk.WriteImage(moving_resampled_test, "/tmp/resamp.nii.gz")
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform_v11, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    return final_transform_v11, moving_resampled
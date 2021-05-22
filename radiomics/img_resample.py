
import logging

import numpy as np
import pywt
import SimpleITK as sitk
import six
from six.moves import range
import yaml

logger = logging.getLogger(__name__)

def cropToTumorMask(imageNode, maskNode, boundingBox, **kwargs):
    """
    Create a sitkImage of the segmented region of the image based on the input label.

    Create a sitkImage of the labelled region of the image, cropped to have a
    cuboid shape equal to the ijk boundaries of the label.

    :param boundingBox: The bounding box used to crop the image. This is the bounding box as returned by
    :py:func:`checkMask`.
    :param label: [1], value of the label, onto which the image and mask must be cropped.
    :return: Cropped image and mask (SimpleITK image instances).

    """
    global logger
    padDistance = kwargs.get('padDistance', 0)

    size = np.array(maskNode.GetSize())

    ijkMinBounds = boundingBox[0::2] - padDistance
    ijkMaxBounds = size - boundingBox[1::2] - padDistance - 1

    # Ensure cropped area is not outside original image bounds
    ijkMinBounds = np.maximum(ijkMinBounds, 0)
    ijkMaxBounds = np.maximum(ijkMaxBounds, 0)

    # Crop Image
    logger.debug('Cropping to size %s', (boundingBox[1::2] - boundingBox[0::2]) + 1)
    cif = sitk.CropImageFilter()
    try:
        cif.SetLowerBoundaryCropSize(ijkMinBounds)
        cif.SetUpperBoundaryCropSize(ijkMaxBounds)
    except TypeError:
        # newer versions of SITK/python want a tuple or list
        cif.SetLowerBoundaryCropSize(ijkMinBounds.tolist())
        cif.SetUpperBoundaryCropSize(ijkMaxBounds.tolist())
    croppedImageNode = cif.Execute(imageNode)
    croppedMaskNode = cif.Execute(maskNode)

    return croppedImageNode, croppedMaskNode

def _checkROI(imageNode, maskNode, **kwargs):
    """
    Check whether maskNode contains a valid ROI defined by label:

    1. Check whether the label value is present in the maskNode.
    2. Check whether the ROI defined by the label does not include an area outside the physical area of the image.

    For the second check, a tolerance of 1e-3 is allowed.

    If the ROI is valid, the bounding box (lower bounds, followd by size in all dimensions (X, Y, Z ordered)) is
    returned. Otherwise, a ValueError is raised.
    """
    label = kwargs.get('label', 1)

    # logger.debug('Checking ROI validity')

    # Determine bounds of cropped volume in terms of original Index coordinate space
    lssif = sitk.LabelShapeStatisticsImageFilter()
    lssif.Execute(maskNode)

    logger.debug('Checking if label %d is persent in the mask', label)
    if label not in lssif.GetLabels():
        raise ValueError('Label (%d) not present in mask', label)

    # LBound and size of the bounding box, as (L_X, L_Y, [L_Z], S_X, S_Y, [S_Z])
    # bb = np.array(lssif.GetBoundingBox(label))
    bb = np.array((0, 0, 0) + maskNode.GetSize())
    print(f'Bounding Box:{bb}')
    Nd = maskNode.GetDimension()
    print(f'Mask Dimension:{Nd}')

    # Determine if the ROI is within the physical space of the image

    logger.debug('Comparing physical space of bounding box to physical space of image')
    # Step 1: Get the origin and UBound corners of the bounding box in physical space
    # The additional 0.5 represents the difference between the voxel center and the voxel corner
    # Upper bound index of ROI = bb[:Nd] + bb[Nd:] - 1 (LBound + Size - 1), .5 is added to get corner
    ROIBounds = (maskNode.TransformContinuousIndexToPhysicalPoint(bb[:Nd] - .5),    # Origin
                             maskNode.TransformContinuousIndexToPhysicalPoint(bb[:Nd] + bb[Nd:] - 0.5))    # UBound
    print(f'Physical Space:{ROIBounds}')
    # Step 2: Translate the ROI physical bounds to the image coordinate space
    ROIBounds = (imageNode.TransformPhysicalPointToContinuousIndex(ROIBounds[0]),    # Origin
                             imageNode.TransformPhysicalPointToContinuousIndex(ROIBounds[1]))
    print(f'Voxel Space:{ROIBounds}')
    # logger.debug('ROI bounds (image coordinate space): %s', ROIBounds)

    # Check if any of the ROI bounds are outside the image indices (i.e. -0.5 < ROI < Im.Size -0.5)
    # The additional 0.5 is to allow for different spacings (defines the edges, not the centers of the edge-voxels
    # Define a tolerance to correct for machine precision errors
    # tolerance = 1e-3
    # if np.any(np.min(ROIBounds, axis=0) < (- .5 - tolerance)) or \
    #      np.any(np.max(ROIBounds, axis=0) > (np.array(imageNode.GetSize()) - .5 + tolerance)):
    #     raise ValueError('Bounding box of ROI is larger than image space:\n\t'
    #                                      'ROI bounds (x, y, z image coordinate space) %s\n\tImage Size %s' %
    #                                      (ROIBounds, imageNode.GetSize()))

    # logger.debug('ROI valid, calculating resampling grid')

    return bb

def resampleImage(imageNode, maskNode, **kwargs):
    """
    Resamples image and mask to the specified pixel spacing (The default interpolator is Bspline).

    Resampling can be enabled using the settings 'interpolator' and 'resampledPixelSpacing' in the parameter file or as
    part of the settings passed to the feature extractor. See also
    :ref:`feature extractor <radiomics-featureextrator-label>`.

    'imageNode' and 'maskNode' are SimpleITK Objects, and 'resampledPixelSpacing' is the output pixel spacing (sequence of
    3 elements).

    If only in-plane resampling is required, set the output pixel spacing for the out-of-plane dimension (usually the last
    dimension) to 0. Spacings with a value of 0 are replaced by the spacing as it is in the original mask.

    Only part of the image and labelmap are resampled. The resampling grid is aligned to the input origin, but only voxels
    covering the area of the image ROI (defined by the bounding box) and the padDistance are resampled. This results in a
    resampled and partially cropped image and mask. Additional padding is required as some filters also sample voxels
    outside of segmentation boundaries. For feature calculation, image and mask are cropped to the bounding box without
    any additional padding, as the feature classes do not need the gray level values outside the segmentation.

    The resampling grid is calculated using only the input mask. Even when image and mask have different directions, both
    the cropped image and mask will have the same direction (equal to direction of the mask). Spacing and size are
    determined by settings and bounding box of the ROI.

    .. note::
      Before resampling the bounds of the non-padded ROI are compared to the bounds. If the ROI bounding box includes
      areas outside of the physical space of the image, an error is logged and (None, None) is returned. No features will
      be extracted. This enables the input image and mask to have different geometry, so long as the ROI defines an area
      within the image.

    .. note::
      The additional padding is adjusted, so that only the physical space within the mask is resampled. This is done to
      prevent resampling outside of the image. Please note that this assumes the image and mask to image the same physical
      space. If this is not the case, it is possible that voxels outside the image are included in the resampling grid,
      these will be assigned a value of 0. It is therefore recommended, but not enforced, to use an input mask which has
      the same or a smaller physical space than the image.
    """
    global logger
    resampledPixelSpacing = kwargs['resampledPixelSpacing']
    interpolator = kwargs.get('interpolator', sitk.sitkBSpline)
    padDistance = kwargs.get('padDistance', 5)
    label = kwargs.get('label', 1)

    logger.debug('Resampling image and mask')

    if imageNode is None or maskNode is None:
        raise ValueError('Requires both image and mask to resample')
  
    # image spacing
    maskSpacing = np.array(maskNode.GetSpacing())
    imageSpacing = np.array(imageNode.GetSpacing())

    Nd_resampled = len(resampledPixelSpacing)
    # image dimensions, 3
    Nd_mask = len(maskSpacing)
    # mask dimensions, 3
    assert Nd_resampled == Nd_mask, \
        'Wrong dimensionality (%i-D) of resampledPixelSpacing!, %i-D required' % (Nd_resampled, Nd_mask)

    # If spacing for a direction is set to 0, use the original spacing (enables "only in-slice" resampling)
    logger.debug('Where resampled spacing is set to 0, set it to the original spacing (mask)')
    resampledPixelSpacing = np.array(resampledPixelSpacing)
    resampledPixelSpacing = np.where(resampledPixelSpacing == 0, maskSpacing, resampledPixelSpacing)

    # Check if the maskNode contains a valid ROI. If ROI is valid, the bounding box needed to calculate the resampling
    # grid is returned.
    bb = _checkROI(imageNode, maskNode, **kwargs)

    # Do not resample in those directions where labelmap spans only one slice.
    maskSize = np.array(maskNode.GetSize())
    resampledPixelSpacing = np.where(bb[Nd_mask:] != 1, resampledPixelSpacing, maskSpacing)

    # If current spacing is equal to resampledPixelSpacing, no interpolation is needed
    # Tolerance = 1e-5 + 1e-8*abs(resampledSpacing)
    logger.debug('Comparing resampled spacing to original spacing (image')
    if np.allclose(imageSpacing, resampledPixelSpacing):
        logger.info('New spacing equal to original image spacing, just resampling the mask')

        # Ensure that image and mask geometry match
        rif = sitk.ResampleImageFilter()
        rif.SetReferenceImage(imageNode)
        rif.SetInterpolator(sitk.sitkNearestNeighbor)
        maskNode = rif.Execute(maskNode)

        # re-calculate the bounding box of the mask
        lssif = sitk.LabelShapeStatisticsImageFilter()
        lssif.Execute(maskNode)
        bb = np.array(lssif.GetBoundingBox(label))

        low_up_bb = np.empty(Nd_mask * 2, dtype=int)
        low_up_bb[::2] = bb[:Nd_mask]
        low_up_bb[1::2] = bb[:Nd_mask] + bb[Nd_mask:] - 1
        return cropToTumorMask(imageNode, maskNode, low_up_bb, **kwargs)

    spacingRatio = maskSpacing / resampledPixelSpacing

    # Determine bounds of cropped volume in terms of new Index coordinate space,
    # round down for lowerbound and up for upperbound to ensure entire segmentation is captured (prevent data loss)
    # Pad with an extra .5 to prevent data loss in case of upsampling. For Ubound this is (-1 + 0.5 = -0.5)
    bbNewLBound = np.floor((bb[:Nd_mask] - 0.5) * spacingRatio - padDistance)
    bbNewUBound = np.ceil((bb[:Nd_mask] + bb[Nd_mask:] - 0.5) * spacingRatio + padDistance)

    # Ensure resampling is not performed outside bounds of original image
    maxUbound = np.ceil(maskSize * spacingRatio) - 1
    bbNewLBound = np.where(bbNewLBound < 0, 0, bbNewLBound)
    bbNewUBound = np.where(bbNewUBound > maxUbound, maxUbound, bbNewUBound)

    # Calculate the new size. Cast to int to prevent error in sitk.
    newSize = np.array(bbNewUBound - bbNewLBound + 1, dtype='int').tolist()

    # Determine continuous index of bbNewLBound in terms of the original Index coordinate space
    bbOriginalLBound = bbNewLBound / spacingRatio

    # Origin is located in center of first voxel, e.g. 1/2 of the spacing
    # from Corner, which corresponds to 0 in the original Index coordinate space.
    # The new spacing will be in 0 the new Index coordinate space. Here we use continuous
    # index to calculate where the new 0 of the new Index coordinate space (of the original volume
    # in terms of the original spacing, and add the minimum bounds of the cropped area to
    # get the new Index coordinate space of the cropped volume in terms of the original Index coordinate space.
    # Then use the ITK functionality to bring the continuous index into the physical space (mm)
    newOriginIndex = np.array(.5 * (resampledPixelSpacing - maskSpacing) / maskSpacing)
    newCroppedOriginIndex = newOriginIndex + bbOriginalLBound
    newOrigin = maskNode.TransformContinuousIndexToPhysicalPoint(newCroppedOriginIndex)

    imagePixelType = imageNode.GetPixelID()
    maskPixelType = maskNode.GetPixelID()

    direction = np.array(maskNode.GetDirection())

    logger.info('Applying resampling from spacing %s and size %s to spacing %s and size %s',
                maskSpacing, maskSize, resampledPixelSpacing, newSize)

    try:
        if isinstance(interpolator, six.string_types):
            interpolator = getattr(sitk, interpolator)
    except Exception:
        logger.warning('interpolator "%s" not recognized, using sitkBSpline', interpolator)
        interpolator = sitk.sitkBSpline

    rif = sitk.ResampleImageFilter()

    rif.SetOutputSpacing(resampledPixelSpacing)
    rif.SetOutputDirection(direction)
    rif.SetSize(newSize)
    rif.SetOutputOrigin(newOrigin)

    logger.debug('Resampling image')
    rif.SetOutputPixelType(imagePixelType)
    rif.SetInterpolator(interpolator)
    resampledImageNode = rif.Execute(imageNode)

    logger.debug('Resampling mask')
    rif.SetOutputPixelType(maskPixelType)
    rif.SetInterpolator(sitk.sitkNearestNeighbor)
    resampledMaskNode = rif.Execute(maskNode)

    return resampledImageNode, resampledMaskNode

def normalizeImage(image, **kwargs):
    r"""
    Normalizes the image by centering it at the mean with standard deviation. Normalization is based on all gray values in
    the image, not just those inside the segmentation.

    :math:`f(x) = \frac{s(x - \mu_x)}{\sigma_x}`

    Where:

    - :math:`x` and :math:`f(x)` are the original and normalized intensity, respectively.
    - :math:`\mu_x` and :math:`\sigma_x` are the mean and standard deviation of the image instensity values.
    - :math:`s` is an optional scaling defined by ``scale``. By default, it is set to 1.

    Optionally, outliers can be removed, in which case values for which :math:`x > \mu_x + n\sigma_x` or
    :math:`x < \mu_x - n\sigma_x` are set to :math:`\mu_x + n\sigma_x` and :math:`\mu_x - n\sigma_x`, respectively.
    Here, :math:`n>0` and defined by ``outliers``. This, in turn, is controlled by the ``removeOutliers`` parameter.
    Removal of outliers is done after the values of the image are normalized, but before ``scale`` is applied.
    """
    global logger
    scale = kwargs.get('normalizeScale', 1)
    outliers = kwargs.get('removeOutliers')

    logger.debug('Normalizing image with scale %d', scale)
    image = sitk.Normalize(image)

    if outliers is not None:
        logger.debug('Removing outliers > %g standard deviations', outliers)
        imageArr = sitk.GetArrayFromImage(image)

        imageArr[imageArr > outliers] = outliers
        imageArr[imageArr < -outliers] = -outliers

        newImage = sitk.GetImageFromArray(imageArr)
        newImage.CopyInformation(image)
        image = newImage

    image *= scale

    return image
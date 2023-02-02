import torchio as tio 
from typing import Iterable, Tuple, Union, List, Optional, Sequence, Dict
from numbers import Number
import nibabel as nib 
import numpy as np
from torchio.typing import TypeRangeFloat
from torchio.transforms.transform import TypeMaskingMethod 
from torchio import Subject, Image
import torch 



class SubjectToTensor(object):
    """Transforms TorchIO Subjects into a Python dict and changes axes order from TorchIO to Torch"""
    def __call__(self, subject: Subject):
        return {key: val.data.swapaxes(1,-1) if isinstance(val, Image) else val  for key,val in subject.items()}

class ImageToTensor(object):
    """Transforms TorchIO Image into a Numpy/Torch Tensor and changes axes order from TorchIO [B, C, W, H, D] to Torch [B, C, D, H, W]"""
    def __call__(self, image: Image):
        return image.data.swapaxes(1,-1)

def parse_per_channel(per_channel, channels):
    if isinstance(per_channel, bool):
        if per_channel == True:
            return [(ch,) for ch in range(channels)]
        else:
            return [tuple(ch for ch in range(channels))] 
    else:
        return per_channel 

class ZNormalization(tio.ZNormalization):
    """Add option 'per_channel' to apply znorm for each channel independently and percentiles to clip values first"""
    def __init__(
        self,
        percentiles: TypeRangeFloat = (0, 100),
        per_channel=True,
        masking_method: TypeMaskingMethod = None,
        **kwargs
    ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.percentiles = percentiles
        self.per_channel = per_channel


    def apply_normalization(
        self,
        subject: Subject,
        image_name: str,
        mask: torch.Tensor,
    ) -> None:
        image = subject[image_name]
        per_channel = parse_per_channel(self.per_channel, image.shape[0])

        image.set_data(torch.cat([
            self._znorm(image.data[chs,], mask[chs,], image_name, image.path)
            for chs in per_channel ])
        )
  

    def _znorm(self, image_data, mask, image_name, image_path):
        cutoff = torch.quantile(image_data, torch.tensor(self.percentiles)/100.0)
        torch.clamp(image_data, *cutoff.tolist(), out=image_data)

        standardized = self.znorm(image_data, mask)
        if standardized is None:
            message = (
                'Standard deviation is 0 for masked values'
                f' in image "{image_name}" ({image_path})'
            )
            raise RuntimeError(message)
        return standardized



class RescaleIntensity(tio.RescaleIntensity):
    """Add option 'per_channel' to apply rescale for each channel independently"""
    def __init__(
        self,
        out_min_max: TypeRangeFloat = (0, 1),
        percentiles: TypeRangeFloat = (0, 100),
        masking_method: TypeMaskingMethod = None,
        in_min_max: Optional[Tuple[float, float]] = None,
        per_channel=True, # Bool or List of tuples containing channel indices that should be normalized together 
        **kwargs
    ):
        super().__init__(out_min_max, percentiles, masking_method, in_min_max, **kwargs)
        self.per_channel=per_channel

    def apply_normalization(
            self,
            subject: Subject,
            image_name: str,
            mask: torch.Tensor,
    ) -> None:
        image = subject[image_name]
        per_channel = parse_per_channel(self.per_channel, image.shape[0])
        
        image.set_data(torch.cat([
            self.rescale(image.data[chs,], mask[chs,], image_name)
            for chs in per_channel ])
        )


class Pad(tio.Pad):
    """Fixed version of TorchIO Pad: 
         * Pads with zeros for LabelMaps independent of padding mode (eg. don't pad with mean)
       Changes: 
         * Pads with global (not per axis) 'maximum', 'mean', 'median', 'minimum' if any of these padding modes were selected"""
    def apply_transform(self, subject: Subject) -> Subject:
        assert self.bounds_parameters is not None
        low = self.bounds_parameters[::2]
        for image in self.get_images(subject):
            new_origin = nib.affines.apply_affine(image.affine, -np.array(low))
            new_affine = image.affine.copy()
            new_affine[:3, 3] = new_origin
            kwargs: Dict[str, Union[str, float]]
            if isinstance(self.padding_mode, Number):
                kwargs = {
                    'mode': 'constant',
                    'constant_values': self.padding_mode,
                }
            elif isinstance(image, tio.LabelMap): # FIX 
                kwargs = {
                    'mode': 'constant',
                    'constant_values': 0,
                }
            else:
                if self.padding_mode in ['maximum', 'mean', 'median', 'minimum']:
                    if self.padding_mode == 'maximum':
                        constant_values = image.data.min()
                    elif self.padding_mode == 'mean':
                        constant_values = image.data.to(torch.float).mean().to(image.data.dtype)
                    elif self.padding_mode == 'median':
                        constant_values = image.data.median()
                    elif self.padding_mode == 'minimum':
                        constant_values = image.data.min()
                    kwargs = {
                        'mode': 'constant',
                        'constant_values': constant_values,
                    }
                else:
                    kwargs = {'mode': self.padding_mode}
            pad_params = self.bounds_parameters
            paddings = (0, 0), pad_params[:2], pad_params[2:4], pad_params[4:]
            padded = np.pad(image.data, paddings, **kwargs)  # type: ignore[call-overload]  # noqa: E501
            image.set_data(torch.as_tensor(padded))
            image.affine = new_affine
        return subject


class CropOrPad(tio.CropOrPad):
    """Fixed version of TorchIO CropOrPad: 
         * Pads with zeros for LabelMaps independent of padding mode (eg. don't pad with mean)
       Changes: 
         * Pads with global (not per axis) 'maximum', 'mean', 'median', 'minimum' if any of these padding modes were selected"""
    def apply_transform(self, subject: Subject) -> Subject:
        subject.check_consistent_space()
        padding_params, cropping_params = self.compute_crop_or_pad(subject)
        padding_kwargs = {'padding_mode': self.padding_mode}
        if padding_params is not None:
            pad = Pad(padding_params, **padding_kwargs)
            subject = pad(subject)  # type: ignore[assignment]
        if cropping_params is not None:
            crop = tio.Crop(cropping_params)
            subject = crop(subject)  # type: ignore[assignment]
        return subject
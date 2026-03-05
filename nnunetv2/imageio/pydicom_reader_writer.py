#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from typing import Tuple, Union, List
import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


class PyDicomIO(BaseReaderWriter):
    """
    Reads and writes DICOM (.dcm) images using pydicom.

    Supports both single-frame (2D) and multi-frame (3D) DICOM files.
    Extracts spacing from PixelSpacing/SliceThickness or SharedFunctionalGroupsSequence.
    Supplementary DICOM metadata (patient, study, series info) is preserved in a
    'pydicom_stuff' sub-dict and converted to strings for JSON serialization.

    Segmentation output is written as a simple DICOM with integer label maps (uint8).
    If you need proper DICOM SEG objects, create them in a postprocessing step.
    """
    supported_file_endings = [
        '.dcm',
    ]

    @staticmethod
    def _is_multiframe(dcm: FileDataset) -> bool:
        """Determine if a DICOM dataset is multi-frame (3D) using the NumberOfFrames tag."""
        num_frames = getattr(dcm, 'NumberOfFrames', None)
        if num_frames is not None:
            return int(num_frames) > 1
        return False

    @staticmethod
    def _extract_spacing(dcm: FileDataset, is_3d: bool) -> list:
        """Extract spacing from DICOM metadata.

        For 3D: returns [slice_thickness, pixel_spacing_row, pixel_spacing_col]
        For 2D: returns [max_pixel_spacing * 999, pixel_spacing_row, pixel_spacing_col]
        (the 999 sentinel follows nnUNet's 2D spacing convention)
        """
        try:
            try:
                pixel_spacing = [float(s) for s in dcm.PixelSpacing]
            except AttributeError:
                sfgs = dcm.SharedFunctionalGroupsSequence[0]
                pms = sfgs.PixelMeasuresSequence[0]
                pixel_spacing = [float(s) for s in pms.PixelSpacing]

            if not is_3d:
                max_sp = max(pixel_spacing)
                return [max_sp * 999, pixel_spacing[0], pixel_spacing[1]]

            slice_thickness = None
            for attr in ('SliceThickness', 'SpacingBetweenSlices'):
                val = getattr(dcm, attr, None)
                if val is not None:
                    slice_thickness = float(val)
                    break

            if slice_thickness is None:
                try:
                    sfgs = dcm.SharedFunctionalGroupsSequence[0]
                    pms = sfgs.PixelMeasuresSequence[0]
                    for attr in ('SliceThickness', 'SpacingBetweenSlices'):
                        val = getattr(pms, attr, None)
                        if val is not None:
                            slice_thickness = float(val)
                            break
                except (AttributeError, IndexError):
                    pass

            if slice_thickness is None:
                print("WARNING: Could not extract slice thickness from DICOM. Setting to 1.0.")
                slice_thickness = 1.0

            return [slice_thickness, pixel_spacing[0], pixel_spacing[1]]
        except Exception as e:
            print(f"WARNING: Could not extract spacing from DICOM. Defaulting to 1.0. Error: {e}")
            if is_3d:
                return [1.0, 1.0, 1.0]
            else:
                return [999.0, 1.0, 1.0]

    @staticmethod
    def _extract_supplementary_props(dcm: FileDataset) -> dict:
        """Extract supplementary properties from DICOM metadata as JSON-serializable strings."""
        props = {}
        for prop in [
            "PatientID",
            "PatientName",
            "PatientBirthDate",
            "PatientSex",
            "StudyID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SOPInstanceUID",
            "SOPClassUID",
            "InstanceNumber",
            "Modality",
            "Manufacturer",
            "ManufacturerModelName",
            "BodyPartExamined",
            "PatientPosition",
            "Laterality",
            "StudyDate",
            "StudyTime",
            "SeriesDate",
            "SeriesTime",
            "AcquisitionDate",
            "AcquisitionTime",
            "DeviceSerialNumber",
            "AccessionNumber",
            "ReferringPhysicianName",
        ]:
            if hasattr(dcm, prop):
                props[prop] = str(getattr(dcm, prop))
        return props

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        ending = '.' + image_fnames[0].split('.')[-1]
        assert ending.lower() in self.supported_file_endings, \
            f'Ending {ending} not supported by {self.__class__.__name__}'

        images = []
        spacings_for_nnunet = []
        supplementary_props = None

        for f in image_fnames:
            dcm = pydicom.dcmread(f)
            image = dcm.pixel_array
            is_3d = self._is_multiframe(dcm)

            if is_3d:
                npy_image = image[None]
            else:
                npy_image = image[None, None]

            images.append(npy_image)
            spacings_for_nnunet.append(self._extract_spacing(dcm, is_3d))

            if supplementary_props is None:
                supplementary_props = self._extract_supplementary_props(dcm)

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings_for_nnunet):
            print('ERROR! Not all input images have the same spacing!')
            print('Spacings:')
            print(spacings_for_nnunet)
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        props = {
            'pydicom_stuff': supplementary_props or {},
            'spacing': spacings_for_nnunet[0]
        }
        return np.vstack(images, dtype=np.float32, casting='unsafe'), props

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname,))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        assert seg.ndim == 3, \
            'segmentation must be 3d. If exporting a 2d segmentation, provide it as shape 1,x,y'
        spacing = properties['spacing']
        is_2d = (spacing[0] >= 999)

        if is_2d:
            seg_out = seg[0]
        else:
            seg_out = seg

        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(output_fname, {}, file_meta=file_meta, preamble=b"\0" * 128)

        pydicom_stuff = properties.get('pydicom_stuff', {})
        ds.PatientName = pydicom_stuff.get('PatientName', 'Unknown')
        ds.PatientID = pydicom_stuff.get('PatientID', 'Unknown')
        ds.StudyInstanceUID = pydicom_stuff.get('StudyInstanceUID', generate_uid())
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.Modality = pydicom_stuff.get('Modality', 'OT')
        ds.SpecificCharacterSet = 'ISO_IR 100'

        ds.PixelSpacing = [spacing[1], spacing[2]]
        if not is_2d:
            ds.SpacingBetweenSlices = spacing[0]

        ds.set_pixel_data(
            seg_out.astype('uint8'),
            photometric_interpretation='MONOCHROME2',
            bits_stored=8,
        )
        ds[0x7FE0, 0x0010].VR = 'OB'

        ds.save_as(output_fname, implicit_vr=False)

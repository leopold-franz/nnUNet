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
import os.path
from typing import Tuple, Union, List
import numpy as np
import nnunetv2
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SegmentationStorage,
    generate_uid,
)
from datetime import datetime
from batchgenerators.utilities.file_and_folder_operations import isfile, load_json, save_json, split_path, join


class PyDicomIO(BaseReaderWriter):
    """
    reads and writes DICOM (.dcm) images. Uses pydicom package. Ignores all metadata except spacing!

    Currently only supports uint8 data type for segmentation images.
    """
    supported_file_endings = [
        '.dcm',
    ]

    @staticmethod
    def _extract_spacing(dcm: FileDataset, is_3d=False) -> np.ndarray:
        """Extract pixel spacing and slice thickness from DICOM dataset.

        Args:
            dcm (FileDataset): DICOM dataset to extract resolution from

        Returns:
            np.ndarray: Array containing [pixel_spacing_x, pixel_spacing_y, slice_thickness]

        Raises:
            AttributeError: If resolution information cannot be found in expected fields
        """
        try:
            try:
                # Directly access the pixel spacing and slice thickness
                pixel_spacing = dcm.PixelSpacing
            except AttributeError:
                # Access the functional groups sequence
                shared_functional_groups_sequence = dcm.SharedFunctionalGroupsSequence[
                    0
                ]
                # Access the pixel measures sequence
                pixel_measures_sequence = (
                    shared_functional_groups_sequence.PixelMeasuresSequence[0]
                )
                # Access the pixel spacing
                pixel_spacing = pixel_measures_sequence.PixelSpacing

            # If the image is 2D, return the pixel spacing
            if not is_3d:
                return np.array(pixel_spacing)
            
            # Access the slice thickness
            try:
                # check if dcm has slice thickness attribute
                if hasattr(dcm, 'SliceThickness'):
                    slice_thickness = dcm.SliceThickness
                elif hasattr(dcm, 'SpacingBetweenSlices'):
                    slice_thickness = dcm.SpacingBetweenSlices
                else:
                    raise AttributeError
            except AttributeError:
                # Access the functional groups sequence
                shared_functional_groups_sequence = dcm.SharedFunctionalGroupsSequence[
                    0
                ]
                # Access the pixel measures sequence
                pixel_measures_sequence = (
                    shared_functional_groups_sequence.PixelMeasuresSequence[0]
                )
                # Access the pixel spacing
                if hasattr(pixel_measures_sequence, 'SliceThickness'):
                    slice_thickness = pixel_measures_sequence.SliceThickness
                elif hasattr(pixel_measures_sequence, 'SpacingBetweenSlices'):
                    slice_thickness = pixel_measures_sequence.SpacingBetweenSlices
                else:
                    raise AttributeError
            return np.array([slice_thickness, pixel_spacing[0], pixel_spacing[1]])
        except Exception as e:
            print(f"WARNING: Could not extract spacing information from DICOM file. Setting spacing for all axis to 1.0. Error: {e}")
            if is_3d:
                return np.array([1.0, 1.0, 1.0])
            else:
                return np.array([1.0, 1.0])
        
    @staticmethod
    def _verify_shape(dcm, is_3d=False) -> None:
        """Verify that the shape of the image is consistent with the metadata.

        Args:
            dcm (FileDataset): DICOM dataset to verify shape of

        Raises:
            RuntimeError: If the shape of the image is not consistent with the metadata
        """
        image_shape = dcm.pixel_array.shape
        if not is_3d:
            metadata_shape = (dcm.Rows, dcm.Columns)
        else:
            metadata_shape = (dcm.NumberOfFrames, dcm.Rows, dcm.Columns)
        if image_shape != metadata_shape:
            raise RuntimeError(
                f"Shape of image ({image_shape}) is not consistent with metadata ({metadata_shape})"
            )

    @staticmethod
    def _extract_supplementary_props(dcm: FileDataset) -> dict:
        """Extract supplementary properties from DICOM metadata.
        Extracted properties:
            - Patient ID
            - Patient Name
            - Study Instance UID
            - Series Instance UID
            - SOP Instance UID
            - Instance Number
            - Modality
            - Manufacturer
            - Manufacturer Model Name
            - Body Part Examined
            - Patient Position
            - Laterality
            - Study Date
            - Study Time
            - Series Date
            - Series Time
            - Acquisition Date
            - Acquisition Time


        Args:
            dcm (FileDataset): DICOM dataset to extract properties from

        Returns:
            dict: Dictionary containing supplementary properties
        """
        props = {}
        # Extract supplementary information from DICOM metadata
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
            "AnatomicRegionSequence",
            "StudyDate",
            "StudyTime",
            "SeriesDate",
            "SeriesTime",
            "AcquisitionDate",
            "AcquisitionTime",
            "DeviceSerialNumber",
            "AccessionNumber",
            "ReferringPhysicianName",
            "Manufacturer",
            "ManufacturerModelName",

        ]:
            if hasattr(dcm, prop):
                props[prop] = getattr(dcm, prop)

        return props

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        # figure out file ending used here
        ending = '.' + image_fnames[0].split('.')[-1]
        assert ending.lower() in self.supported_file_endings, f'Ending {ending} not supported by {self.__class__.__name__}'

        images = []
        for f in image_fnames:
            dcm = pydicom.dcmread(f)
            image = dcm.pixel_array
            is_3d = (len(image.shape) == 3) and all([i > 3 for i in image.shape])
            # verify shape of image corresponds to metadata shape
            self._verify_shape(dcm, is_3d)
            images.append(image[None])
            spacing = self._extract_spacing(dcm, is_3d=True)
            props = {'spacing': spacing.tolist()}
            # Extract supplimentary information from DICOM metadata
            props.update(self._extract_supplementary_props(dcm))

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()

        return np.vstack(images, dtype=np.float32, casting='unsafe'), props
    
    def setup_segmentation_dcm(
        self,
        properties: dict,
        segmentation_array: np.ndarray,
    ) -> FileDataset:
        """Create a DICOM segmentation object from a segmentation array.

        Parameters
        ----------
        properties : dict
            Dictionary containing the properties of the segmentation object

        Returns
        -------
        pydicom.dataset.FileDataset
            The DICOM segmentation object
        """
        
        # Create a new DICOM dataset
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = SegmentationStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        # Create the FileDataset
        ds = FileDataset(
            "segmentation.dcm",
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )

        # Add mandatory patient and study information
        ds.PatientName = properties.get("PatientName", "Unknown")
        ds.PatientID = properties.get("PatientID", "Unknown")
        ds.StudyInstanceUID = properties.get("StudyInstanceUID", generate_uid())
        ds.InstanceNumber = properties.get("InstanceNumber", 1)
        ds.StudyDate = properties.get("StudyDate", "20000101")
        ds.StudyTime = properties.get("StudyTime", "000000")

        # Add extra patient and study information if available
        if "PatientBirthDate" in properties.keys():
            ds.PatientBirthDate = properties["PatientBirthDate"]
        if "PatientSex" in properties.keys():
            ds.PatientSex = properties["PatientSex"]
        if "StudyID" in properties.keys():
            ds.StudyID = properties["StudyID"]

        # Copy location tags from original DICOM
        if "BodyPartExamined" in properties.keys():
            ds.BodyPartExamined = properties["BodyPartExamined"]

        if "Laterality" in properties.keys():
            ds.Laterality = properties["Laterality"]

        if "AnatomicRegionSequence" in properties.keys():
            ds.AnatomicRegionSequence = properties["AnatomicRegionSequence"]

        # Add missing tags for the modules
        # Enhanced General Equipment
        ds.Manufacturer = properties.get("Manufacturer", "Unknown")
        ds.ManufacturerModelName = properties.get("ManufacturerModelName", "Unknown")
        if "DeviceSerialNumber" in properties.keys():
            ds.DeviceSerialNumber = properties["DeviceSerialNumber"]
        else:
            ds.DeviceSerialNumber = "123456"
        ds.SoftwareVersions = "nnUNetv2_" #TODO: + nnunetv2.__version__

        # General Series
        ds.SeriesNumber = 2  # Assuming a placeholder series number

        # General Study
        if "AccessionNumber" in properties.keys():
            ds.AccessionNumber = properties["AccessionNumber"]
        if "ReferringPhysicianName" in properties.keys():
            ds.ReferringPhysicianName = properties["ReferringPhysicianName"]

        # Multi-frame Dimension
        # Define the Dimension Organization Sequence and Dimension Index Sequence if needed
        ds.DimensionOrganizationSequence = [Dataset()]
        ds.DimensionOrganizationSequence[0].DimensionOrganizationUID = (
            generate_uid()
        )
        ds.DimensionIndexSequence = [Dataset()]
        ds.DimensionIndexSequence[0].DimensionOrganizationUID = (
            ds.DimensionOrganizationSequence[0].DimensionOrganizationUID
        )
        ds.DimensionIndexSequence[0].DimensionIndexPointer = (
            0x0020,
            0x9157,
        )
        ds.DimensionIndexSequence[0].FunctionalGroupPointer = (
            0x0020,
            0x9111,
        )

        # Multi-frame Functional Groups
        # Define the Shared Functional Groups Sequence
        ds.SharedFunctionalGroupsSequence = [Dataset()]

        # Segmentation Image
        ds.LossyImageCompression = "00"
        ds.SegmentationType = "BINARY"
        ds.ContentLabel = "SEG"
        ds.ContentDescription = "Segmentation" # Add a description if needed

        # Segmentation Series
        ds.SeriesNumber = 1  # Assuming a placeholder series number

        ds.SeriesInstanceUID = generate_uid(
            entropy_srcs=[
                properties["PatientID"],
                properties["StudyInstanceUID"],
                "Segmentation",
            ]
        )

        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "SEG"
        ds.SOPClassUID = SegmentationStorage

        # Add other required attributes
        ds.SpecificCharacterSet = "ISO_IR 100"
        ds.ImageType = ["DERIVED", "PRIMARY"]
        ds.ContentDate = datetime.now().strftime("%Y%m%d")
        ds.ContentTime = datetime.now().strftime("%H%M%S")

        # Define the segments
        # Create single unknown segment sequence for now
        segment = Dataset()
        segment.SegmentNumber = 1
        segment.SegmentLabel = "Unknown"
        ds.SegmentSequence = [segment]


        # Create Referenced Series Sequence
        ref_series_seq = Dataset()
        ref_series_seq.SeriesInstanceUID = properties["SeriesInstanceUID"]
        ref_series_seq.ReferencedInstanceSequence = []

        # Reference the original 3D instance
        ref_instance = Dataset()
        ref_instance.ReferencedSOPClassUID = properties["SOPClassUID"]
        ref_instance.ReferencedSOPInstanceUID = properties["SOPInstanceUID"]
        ref_series_seq.ReferencedInstanceSequence.append(ref_instance)

        # Add the Referenced Series Sequence to the dataset
        ds.ReferencedSeriesSequence = [ref_series_seq]

        # Add the necessary DICOM tags for segmentation
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        if segmentation_array.ndim == 3:
            ds.NumberOfFrames = segmentation_array.shape[0]
            ds.SpacingBetweenSlices = properties["spacing"][0]
            ds.PixelSpacing = properties["spacing"][1:]
        else:
            ds.NumberOfFrames = 1
            ds.PixelSpacing = properties["spacing"]
        ds.Rows = segmentation_array.shape[1]
        ds.Columns = segmentation_array.shape[2]
        ds.PixelRepresentation = 0

        # Save numpy array to pixel array
        data_bytes = segmentation_array.astype(np.uint8).tobytes()
        if len(data_bytes) % 2 != 0:
            data_bytes += b"\x00"  # Pad with a zero byte to make the length even
        ds.PixelData = data_bytes

        # Update the PixelData VR to 'OB' for 8-bit data
        ds[0x7FE0, 0x0010].VR = "OB"

        return ds

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        # Squeeze the segmentation array to remove the channel dimension if present
        seg = np.squeeze(seg)
        # Create a new DICOM dataset
        dcm = self.setup_segmentation_dcm(properties, seg)
        # Save the DICOM dataset to a file
        dcm.save_as(output_fname, implicit_vr=False)

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        # figure out file ending used here
        ending = '.' + seg_fname.split('.')[-1]
        assert ending.lower() in self.supported_file_endings, f'Ending {ending} not supported by {self.__class__.__name__}'

        seg_dcm = pydicom.dcmread(seg_fname)
        seg = seg_dcm.pixel_array
        is_3d = (len(seg.shape) == 3) and all([i > 3 for i in seg.shape])
        spacing = self._extract_spacing(seg_dcm, is_3d=is_3d)
        
        # add channel dimension
        seg = seg[None]
        return seg.astype(np.float32, copy=False), {'spacing': spacing.tolist()}

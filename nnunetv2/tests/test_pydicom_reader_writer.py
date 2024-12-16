import unittest
import numpy as np
import pydicom
import os
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import SegmentationStorage, ExplicitVRLittleEndian
from pydicom.uid import generate_uid
from nnunetv2.imageio.pydicom_reader_writer import PyDicomIO

class TestPyDicomIO(unittest.TestCase):

    def setUp(self):
        # Create a sample DICOM dataset for testing
        # Create a new DICOM dataset
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = SegmentationStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        # Create the FileDataset
        self.dcm = FileDataset(
            "US_test_image.dcm",
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128,
        )
        self.dcm.PixelSpacing = [0.5, 0.5]
        self.dcm.SliceThickness = 1.0
        self.dcm.Rows = 256
        self.dcm.Columns = 256
        self.dcm.NumberOfFrames = 10
        self.dcm.PixelData = np.random.randint(0, 256, (10, 256, 256), dtype=np.uint8).tobytes()
        self.dcm.PatientID = "12345"
        self.dcm.PatientName = "Test^Patient"
        self.dcm.StudyInstanceUID = "1.2.3.4"
        self.dcm.SeriesInstanceUID = "1.2.3.4.5"
        self.dcm.SOPInstanceUID = "1.2.3.4.5.6"
        self.dcm.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"

    def test_extract_spacing(self):
        spacing = PyDicomIO._extract_spacing(self.dcm, is_3d=True)
        np.testing.assert_array_equal(spacing, [1.0, 0.5, 0.5])

    def test_verify_shape(self):
        PyDicomIO._verify_shape(self.dcm, is_3d=True)
        self.assertRaises(RuntimeError, PyDicomIO._verify_shape, self.dcm, is_3d=False)

    def test_extract_supplementary_props(self):
        props = PyDicomIO._extract_supplementary_props(self.dcm)
        self.assertEqual(props["PatientID"], "12345")
        self.assertEqual(props["PatientName"], "Test^Patient")

    def test_read_images(self):
        # Save the sample DICOM dataset to a file
        self.dcm.save_as("test.dcm")
        reader = PyDicomIO()
        images, props = reader.read_images(["test.dcm"])
        self.assertEqual(images.shape, (10, 1, 256, 256))
        self.assertEqual(props["spacing"], [1.0, 0.5, 0.5])
        os.remove("test.dcm")

    def test_write_seg(self):
        seg = np.random.randint(0, 2, (10, 256, 256), dtype=np.uint8)
        properties = {
            "PatientID": "12345",
            "StudyInstanceUID": "1.2.3.4",
            "SeriesInstanceUID": "1.2.3.4.5",
            "SOPInstanceUID": "1.2.3.4.5.6",
            "SOPClassUID": "1.2.840.10008.5.1.4.1.1.2",
            "spacing": [1.0, 0.5, 0.5]
        }
        writer = PyDicomIO()
        writer.write_seg(seg, "test_seg.dcm", properties)
        seg_dcm = pydicom.dcmread("test_seg.dcm")
        self.assertEqual(seg_dcm.Rows, 256)
        self.assertEqual(seg_dcm.Columns, 256)
        self.assertEqual(seg_dcm.NumberOfFrames, 10)
        os.remove("test_seg.dcm")

    def test_read_seg(self):
        seg = np.random.randint(0, 2, (128, 128, 128), dtype=np.uint8)
        properties = {
            "PatientID": "12345",
            "StudyInstanceUID": "1.2.3.4",
            "SeriesInstanceUID": "1.2.3.4.5",
            "SOPInstanceUID": "1.2.3.4.5.6",
            "SOPClassUID": "1.2.840.10008.5.1.4.1.1.2",
            "spacing": [1.0, 0.5, 0.5]
        }
        writer = PyDicomIO()
        writer.write_seg(seg, "test_seg.dcm", properties)
        reader = PyDicomIO()
        read_seg, props = reader.read_seg("test_seg.dcm")
        self.assertEqual(read_seg.shape, (128, 128, 128))
        self.assertEqual(props["spacing"], [1.0, 0.5, 0.5])
        os.remove("test_seg.dcm")

if __name__ == "__main__":
    unittest.main()

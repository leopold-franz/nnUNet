import copy
import os
import tempfile
import unittest

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from nnunetv2.imageio.pydicom_reader_writer import PyDicomIO


class TestPyDicomIO(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmpdir.name

        # --- 3D multi-frame DICOM ---
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        self.dcm_3d = FileDataset(
            "test_3d.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128,
        )
        self.dcm_3d.PixelSpacing = [0.5, 0.5]
        self.dcm_3d.SliceThickness = 1.0
        self.dcm_3d.PatientID = "12345"
        self.dcm_3d.PatientName = "Test^Patient"
        self.dcm_3d.StudyInstanceUID = "1.2.3.4"
        self.dcm_3d.SeriesInstanceUID = "1.2.3.4.5"
        self.dcm_3d.SOPInstanceUID = generate_uid()
        self.dcm_3d.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        self.dcm_3d.Modality = "US"
        arr_3d = np.random.randint(0, 256, (10, 256, 256), dtype=np.uint8)
        self.dcm_3d.set_pixel_data(
            arr_3d, photometric_interpretation="MONOCHROME2", bits_stored=8,
        )

        # --- 2D single-frame DICOM ---
        file_meta_2d = Dataset()
        file_meta_2d.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
        file_meta_2d.MediaStorageSOPInstanceUID = generate_uid()
        file_meta_2d.TransferSyntaxUID = ExplicitVRLittleEndian

        self.dcm_2d = FileDataset(
            "test_2d.dcm", {}, file_meta=file_meta_2d, preamble=b"\0" * 128,
        )
        self.dcm_2d.PixelSpacing = [0.3, 0.3]
        self.dcm_2d.PatientID = "67890"
        self.dcm_2d.PatientName = "Test^Patient2D"
        self.dcm_2d.StudyInstanceUID = "5.6.7.8"
        self.dcm_2d.SeriesInstanceUID = "5.6.7.8.9"
        self.dcm_2d.SOPInstanceUID = generate_uid()
        self.dcm_2d.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        self.dcm_2d.Modality = "US"
        arr_2d = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        self.dcm_2d.set_pixel_data(
            arr_2d, photometric_interpretation="MONOCHROME2", bits_stored=8,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    # --- _is_multiframe ---

    def test_is_multiframe_3d(self):
        self.assertTrue(PyDicomIO._is_multiframe(self.dcm_3d))

    def test_is_multiframe_2d(self):
        self.assertFalse(PyDicomIO._is_multiframe(self.dcm_2d))

    # --- _extract_spacing ---

    def test_extract_spacing_3d(self):
        spacing = PyDicomIO._extract_spacing(self.dcm_3d, is_3d=True)
        self.assertEqual(spacing, [1.0, 0.5, 0.5])

    def test_extract_spacing_2d(self):
        spacing = PyDicomIO._extract_spacing(self.dcm_2d, is_3d=False)
        self.assertAlmostEqual(spacing[0], 0.3 * 999)
        self.assertAlmostEqual(spacing[1], 0.3)
        self.assertAlmostEqual(spacing[2], 0.3)

    # --- _extract_supplementary_props ---

    def test_supplementary_props_are_strings(self):
        props = PyDicomIO._extract_supplementary_props(self.dcm_3d)
        self.assertEqual(props["PatientID"], "12345")
        self.assertEqual(props["PatientName"], "Test^Patient")
        for key, value in props.items():
            self.assertIsInstance(value, str, f"Property {key} should be str, got {type(value)}")

    # --- read_images ---

    def test_read_images_3d(self):
        path = os.path.join(self.tmp_path, "test_3d.dcm")
        self.dcm_3d.save_as(path, implicit_vr=False)
        images, props = PyDicomIO().read_images([path])
        self.assertEqual(images.shape, (1, 10, 256, 256))
        self.assertEqual(props["spacing"], [1.0, 0.5, 0.5])
        self.assertIn("pydicom_stuff", props)
        self.assertEqual(props["pydicom_stuff"]["PatientID"], "12345")

    def test_read_images_2d(self):
        path = os.path.join(self.tmp_path, "test_2d.dcm")
        self.dcm_2d.save_as(path, implicit_vr=False)
        images, props = PyDicomIO().read_images([path])
        self.assertEqual(images.shape, (1, 1, 128, 128))
        self.assertAlmostEqual(props["spacing"][0], 0.3 * 999)

    def test_read_images_spacing_consistency(self):
        path1 = os.path.join(self.tmp_path, "img1.dcm")
        self.dcm_3d.save_as(path1, implicit_vr=False)

        dcm2 = copy.deepcopy(self.dcm_3d)
        dcm2.PixelSpacing = [1.0, 1.0]
        path2 = os.path.join(self.tmp_path, "img2.dcm")
        dcm2.save_as(path2, implicit_vr=False)

        with self.assertRaises(RuntimeError):
            PyDicomIO().read_images([path1, path2])

    # --- write_seg ---

    def test_write_seg_3d(self):
        seg = np.random.randint(0, 2, (10, 256, 256), dtype=np.uint8)
        properties = {
            "pydicom_stuff": {"PatientID": "12345", "Modality": "US"},
            "spacing": [1.0, 0.5, 0.5],
        }
        path = os.path.join(self.tmp_path, "seg_3d.dcm")
        PyDicomIO().write_seg(seg, path, properties)
        dcm = pydicom.dcmread(path)
        self.assertEqual(dcm.Rows, 256)
        self.assertEqual(dcm.Columns, 256)
        self.assertEqual(int(dcm.NumberOfFrames), 10)

    def test_write_seg_2d(self):
        seg = np.random.randint(0, 2, (1, 128, 128), dtype=np.uint8)
        properties = {
            "pydicom_stuff": {"PatientID": "12345"},
            "spacing": [999.0, 0.3, 0.3],
        }
        path = os.path.join(self.tmp_path, "seg_2d.dcm")
        PyDicomIO().write_seg(seg, path, properties)
        dcm = pydicom.dcmread(path)
        self.assertEqual(dcm.Rows, 128)
        self.assertEqual(dcm.Columns, 128)
        self.assertFalse(hasattr(dcm, 'NumberOfFrames'))

    # --- round-trip tests ---

    def test_round_trip_3d(self):
        seg_orig = np.random.randint(0, 3, (10, 128, 128), dtype=np.uint8)
        properties = {
            "pydicom_stuff": {"PatientID": "RT3D"},
            "spacing": [1.0, 0.5, 0.5],
        }
        path = os.path.join(self.tmp_path, "rt3d.dcm")
        io = PyDicomIO()
        io.write_seg(seg_orig, path, properties)
        seg_read, props_read = io.read_seg(path)
        np.testing.assert_array_equal(seg_read[0].astype(np.uint8), seg_orig)
        self.assertEqual(props_read["spacing"], [1.0, 0.5, 0.5])

    def test_round_trip_2d(self):
        seg_orig = np.random.randint(0, 3, (1, 64, 64), dtype=np.uint8)
        properties = {
            "pydicom_stuff": {},
            "spacing": [999.0, 1.0, 1.0],
        }
        path = os.path.join(self.tmp_path, "rt2d.dcm")
        io = PyDicomIO()
        io.write_seg(seg_orig, path, properties)
        seg_read, props_read = io.read_seg(path)
        # read_seg on a 2D DICOM returns shape (1, 1, y, x)
        np.testing.assert_array_equal(seg_read[0, 0].astype(np.uint8), seg_orig[0])
        self.assertAlmostEqual(props_read["spacing"][0], 999.0)


if __name__ == "__main__":
    unittest.main()

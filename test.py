
import fct as f
import pydicom
steps = f.make_step(300,3)
print(steps)
import numpy as np
reader = pydicom.dcmread("MR_small.dcm")
array = reader.pixel_array
a = np.array([[1,2,3j],[4,5j+2,6],[7,8+1j,9]])
b = abs(a)
print("AAAAAAAAAAAA")
print(a)
print("BBBBBBBBBBBBBB")
print(b)
k = b.reshape(-1)

print(k)
gabor_list = f.gabor_blank_filter(39,5,8)
print("SSSSSSSSAMAMAMAMAMAOOOOOOOOOOORRRRRr")
print(gabor_list[0][0])
# feature_size= 40
# gabor_fet = f.gabor_feature(array, gabor_list, 1, 1)
#
#
# c = np.array([[5,5,0],[9,6,0.2],[1,0.1,0]])
# decompositon = f.gabor_decomposition(, 5, 8)
# print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
# print(len(gabor_fet))
# print((decompositon[:,:,0]))
# k = np.minimum(c, 0.5) * 10
# print(k)



# k = f.gabor_decomposition(array, 5,8)
# print(k[0])

# mutual_info_left, B = f.mutual_information_gabor_lowest_intensity(array, 300,700,452,3987,3,6)
# print("mutual_info_left")
# print(mutual_info_left)
# print("B")
# print(B)













# from Service.GRAIL import Data
# from Service.PreProcessService import PreProcess
# import matplotlib.pyplot as plt
# from Mapper.mathOperation import PixelArrayOperation
# import numpy as np
# import dicom, dicom.UID
# import numpy as np
# import datetime, time
# import pydicom
# from pydicom.dataset import Dataset, FileDataset
# from pydicom.uid import ExplicitVRLittleEndian
# import pydicom._storage_sopclass_uids
#
# def write_dicom(pixel_array,filename):
#     """
#     INPUTS:
#     pixel_array: 2D numpy ndarray.  If pixel_array is larger than 2D, errors.
#     filename: string name for the output file.
#     """
#
#     ## This code block was taken from the output of a MATLAB secondary
#     ## capture.  I do not know what the long dotted UIDs mean, but
#     ## this code works.
#     pixel_array = pixel_array.astype(np.uint16)
#
#     print("Setting file meta information...")
#     # Populate required values for file meta information
#
#     meta = pydicom.Dataset()
#     meta.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
#     meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
#     meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
#
#     ds = Dataset()
#     ds.file_meta = meta
#
#     ds.is_little_endian = True
#     ds.is_implicit_VR = False
#
#     ds.SOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
#     ds.PatientName = "Test^Firstname"
#     ds.PatientID = "123456"
#
#     ds.Modality = "MR"
#     ds.SeriesInstanceUID = pydicom.uid.generate_uid()
#     ds.StudyInstanceUID = pydicom.uid.generate_uid()
#     ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
#
#     ds.BitsStored = 16
#     ds.BitsAllocated = 16
#     ds.SamplesPerPixel = 1
#     ds.HighBit = 15
#
#     ds.ImagesInAcquisition = "1"
#
#     ds.Rows = pixel_array.shape[0]
#     ds.Columns = pixel_array.shape[1]
#     ds.InstanceNumber = 1
#
#     ds.ImagePositionPatient = r"0\0\1"
#     ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
#     ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"
#
#     ds.RescaleIntercept = "0"
#     ds.RescaleSlope = "1"
#     ds.PixelSpacing = r"1\1"
#     ds.PhotometricInterpretation = "MONOCHROME2"
#     ds.PixelRepresentation = 1
#
#     pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
#
#     print("Setting pixel data...")
#     ds.PixelData = pixel_array.tobytes()
#     ds.save_as(filename)
#     return
#
# import logging
#
# # Create and configure logger
# logging.basicConfig(filename="newfile.log",
#                     format='%(asctime)s %(message)s',
#                     filemode='w')
#
# # Creating an object
# logger = logging.getLogger()
#
# # Setting the threshold of logger to DEBUG
# logger.setLevel(logging.DEBUG)
#
#
# import matplotlib.pyplot as plt
# g = Data("test.dcm")
# pixelData = g.get_pixel_data()
# logger.debug("Pixel Data \n {}".format(pixelData))
# logger.debug("Pixel Data shape \n {}".format(pixelData.shape))
#
# x = g.main()
# plt.imshow(pixelData,cmap=plt.cm.gray)
# plt.show()
# plt.imshow(x,cmap=plt.cm.gray)
# plt.show()
#
# # x = np.arange(16).reshape(16,1)
# # print(x)
# # pixel_array = (x + x.T) * 32
# # pixel_array = np.tile(pixel_array,(16,16))
# # write_dicom(pixel_array,'pretty.dcm')
#
# # dRead = pydicom.dcmread("temp.dcm",force=True)
# # pixelData = dRead.pixel_array
# # dimensions = dRead.pixel_array.shape
# # print(dimensions)
# # plt.imshow(pixelData, cmap=plt.cm.gray)
# # plt.show()
# # out = PixelArrayOperation.from12bitTo8bit(pixelData,1928,4090)
# # plt.imshow(out, cmap=plt.cm.gray)
# # plt.show()
#
# #
# # # img = cv.imread('bob.png', cv.IMREAD_COLOR)
# # # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # # uint8_image = np.uint8(pixelData)
# # # # mein norrmalization
# # # normalized_image = Normalization.normalize(uint8_image)
# # # # cv2 normalized_image
# # # out = np.zeros(dimensions)
# # # cv2_normalize = cv.normalize(uint8_image, out, 0, 255, cv.NORM_MINMAX)
# #
# # # uint8_image = np.uint8(normalized_image)
# # # uint16_image=np.uint16(uint8_image)
# #
# #
# # # max = 2000
# # # test_img = PreProcess.OtsuThresholding(pixelData, max)
# #
# # gamma = 3
# # test_img = PreProcess.GammaCorrection(pixelData,gamma)
# #
# # brt = 100
# # ctrst = 3000
# # test_img = PreProcess.ContrastAdjust(pixelData,ctrst,brt)
# #
# #
# # # print(img)
# # # cv.imshow("hello",test_img)
# # print("pixel data max = ")
# # print(pixelData.max())
# # print("Gamma max = ")
# # print(test_img.max())
# # plt.imshow(pixelData)
# # plt.show()
# # plt.imshow(test_img)
# # plt.show()
#
#
# from Mapper.mathOperation import InformationTheory as TI
#
# data = np.array([2,8,5,18])
# ind = np.array([9,8,5,3])
#
# print("L'entropie de data ")
# print(TI.entropy(data))
# print("L'entropie de ind ")
# print(TI.entropy(ind))
# print("L'entropie conjointe de data et ind ")
# print(TI.joint_entropy(data,ind))
# print("MI is")
# print(TI.mutual_information(data,ind))
import os
import sys
import shutil
import time
import subprocess
import multiprocessing

import cv2
import dlib
import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from skimage import transform as trans

from models import create_model
from util.visualizer import save_crop
from data.image_folder import make_dataset
from options.test_options import TestOptions

sys.path.append("FaceLandmarkDetection")
import face_alignment


###########################################################################
################# functions of crop and align face images #################
###########################################################################
def get_5_points(img):
    dets = detector(img, 1)
    if len(dets) == 0:
        return None
    areas = []
    if len(dets) > 1:
        print(
            "\t###### Warning: more than one face is detected. In this version, we only handle the largest one."
        )
    for i in range(len(dets)):
        area = (dets[i].rect.right() - dets[i].rect.left()) * (
            dets[i].rect.bottom() - dets[i].rect.top()
        )
        areas.append(area)
    ins = areas.index(max(areas))
    shape = sp(img, dets[ins].rect)
    single_points = []
    for i in range(5):
        single_points.append([shape.part(i).x, shape.part(i).y])
    return np.array(single_points)


def align_and_save(
    img_path, save_path, save_input_path, save_param_path, upsample_scale=2
):
    out_size = (512, 512)
    img = dlib.load_rgb_image(img_path)
    h, w, _ = img.shape
    source = get_5_points(img)
    if source is None:  #
        print("\t################ No face is detected")
        return
    tform = trans.SimilarityTransform()
    tform.estimate(source, reference)
    M = tform.params[0:2, :]
    crop_img = cv2.warpAffine(img, M, out_size)
    io.imsave(save_path, crop_img)  # save the crop and align face
    if save_input_path:
        io.imsave(save_input_path, img)  # save the whole input image
    tform2 = trans.SimilarityTransform()
    tform2.estimate(reference, source * upsample_scale)
    # inv_M = cv2.invertAffineTransform(M)
    np.savetxt(
        save_param_path, tform2.params[0:2, :], fmt="%.3f"
    )  # save the inverse affine parameters


def reverse_align(input_path, face_path, param_path, save_path, upsample_scale=2):
    out_size = (512, 512)
    input_img = dlib.load_rgb_image(input_path)
    h, w, _ = input_img.shape
    face512 = dlib.load_rgb_image(face_path)
    inv_M = np.loadtxt(param_path)
    inv_crop_img = cv2.warpAffine(
        face512, inv_M, (w * upsample_scale, h * upsample_scale)
    )
    mask = np.ones((512, 512, 3), dtype=np.float32)  # * 255
    inv_mask = cv2.warpAffine(mask, inv_M, (w * upsample_scale, h * upsample_scale))
    upsample_img = cv2.resize(input_img, (w * upsample_scale, h * upsample_scale))
    inv_mask_erosion_removeborder = cv2.erode(
        inv_mask, np.ones((2 * upsample_scale, 2 * upsample_scale), np.uint8)
    )  # to remove the black border
    inv_crop_img_removeborder = inv_mask_erosion_removeborder * inv_crop_img
    total_face_area = np.sum(inv_mask_erosion_removeborder) // 3
    w_edge = (
        int(total_face_area ** 0.5) // 20
    )  # compute the fusion edge based on the area of face
    erosion_radius = w_edge * 2
    inv_mask_center = cv2.erode(
        inv_mask_erosion_removeborder,
        np.ones((erosion_radius, erosion_radius), np.uint8),
    )
    blur_size = w_edge * 2
    inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
    merge_img = (
        inv_soft_mask * inv_crop_img_removeborder + (1 - inv_soft_mask) * upsample_img
    )
    io.imsave(save_path, merge_img.astype(np.uint8))


###########################################################################
################ functions of preparing the test images ###################
###########################################################################
def AddUpSample(img):
    return img.resize((512, 512), Image.BICUBIC)


def get_part_location(partpath, imgname):
    Landmarks = []
    if not os.path.exists(os.path.join(partpath, imgname + ".txt")):
        print(os.path.join(partpath, imgname + ".txt"))
        print("\t################ No landmark file")
        return 0
    with open(os.path.join(partpath, imgname + ".txt"), "r") as f:
        for line in f:
            tmp = [np.float(i) for i in line.split(" ") if i != "\n"]
            Landmarks.append(tmp)
    Landmarks = np.array(Landmarks)
    Map_LE = list(np.hstack((range(17, 22), range(36, 42))))
    Map_RE = list(np.hstack((range(22, 27), range(42, 48))))
    Map_NO = list(range(29, 36))
    Map_MO = list(range(48, 68))
    try:
        # left eye
        Mean_LE = np.mean(Landmarks[Map_LE], 0)
        L_LE = np.max(
            (
                np.max(np.max(Landmarks[Map_LE], 0) - np.min(Landmarks[Map_LE], 0)) / 2,
                16,
            )
        )
        Location_LE = np.hstack((Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
        # right eye
        Mean_RE = np.mean(Landmarks[Map_RE], 0)
        L_RE = np.max(
            (
                np.max(np.max(Landmarks[Map_RE], 0) - np.min(Landmarks[Map_RE], 0)) / 2,
                16,
            )
        )
        Location_RE = np.hstack((Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)
        # nose
        Mean_NO = np.mean(Landmarks[Map_NO], 0)
        L_NO = np.max(
            (
                np.max(np.max(Landmarks[Map_NO], 0) - np.min(Landmarks[Map_NO], 0)) / 2,
                16,
            )
        )
        Location_NO = np.hstack((Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)
        # mouth
        Mean_MO = np.mean(Landmarks[Map_MO], 0)
        L_MO = np.max(
            (
                np.max(np.max(Landmarks[Map_MO], 0) - np.min(Landmarks[Map_MO], 0)) / 2,
                16,
            )
        )
        Location_MO = np.hstack((Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)
    except:
        return 0
    return (
        torch.from_numpy(Location_LE).unsqueeze(0),
        torch.from_numpy(Location_RE).unsqueeze(0),
        torch.from_numpy(Location_NO).unsqueeze(0),
        torch.from_numpy(Location_MO).unsqueeze(0),
    )


def obtain_inputs(img_path, Landmark_path, img_name):
    A_paths = os.path.join(img_path, img_name)
    A = Image.open(A_paths).convert("RGB")
    Part_locations = get_part_location(Landmark_path, img_name)
    if Part_locations == 0:
        return 0
    C = A
    A = AddUpSample(A)
    A = transforms.ToTensor()(A)
    C = transforms.ToTensor()(C)
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)  #
    C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)  #
    return {
        "A": A.unsqueeze(0),
        "C": C.unsqueeze(0),
        "A_paths": A_paths,
        "Part_locations": Part_locations,
    }


def worker(Params):
    (
        TestImgPath,
        SaveRestorePath,
        SaveParamPath,
        SaveFinalPath,
        UpScaleWhole,
        ImgPath,
    ) = Params
    ImgName = os.path.split(ImgPath)[-1]
    WholeInputPath = os.path.join(TestImgPath, ImgName)
    FaceResultPath = os.path.join(SaveRestorePath, ImgName)
    ParamPath = os.path.join(SaveParamPath, ImgName + ".npy")
    SaveWholePath = os.path.join(SaveFinalPath, ImgName)
    reverse_align(
        WholeInputPath, FaceResultPath, ParamPath, SaveWholePath, UpScaleWhole
    )


if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.which_epoch = "latest"  #

    #######################################################################
    ########################### Test Param ################################
    #######################################################################
    TestImgPath = opt.test_path
    ResultsDir = opt.results_dir
    UpScaleWhole = opt.upscale_factor

    if not TestImgPath or not ResultsDir:
        print(
            'Correct usage: python test_FaceDict.py --test_path "TestData/TestWhole" --results_dir "Results/TestWholeResults" --upscale_factor 4 --gpu_ids -1'
        )
        exit()

    # Create results directory tree.
    SaveInputPath = os.path.join(ResultsDir, "Step0_Input")
    SaveCropPath = os.path.join(ResultsDir, "Step1_CropImg")
    SaveParamPath = os.path.join(ResultsDir, "Step1_AffineParam")
    SaveLandmarkPath = os.path.join(ResultsDir, "Step2_Landmarks")
    SaveRestorePath = os.path.join(ResultsDir, "Step3_RestoreCropFace")
    SaveFinalPath = os.path.join(ResultsDir, "Step4_FinalResults")
    for path in (
        SaveInputPath,
        SaveCropPath,
        SaveParamPath,
        SaveLandmarkPath,
        SaveRestorePath,
        SaveFinalPath,
    ):
        if not os.path.exists(path):
            os.makedirs(path)

    print(
        "\n###################### Now Running the X {} task ##############################".format(
            UpScaleWhole
        )
    )

    # Video input.
    VideoPath = None
    if any(TestImgPath.endswith("." + ext) for ext in ("mp4", "webm", "mkv")):
        assert UpScaleWhole == 1
        VideoPath = TestImgPath
        TestImgPath = SaveInputPath
        if len(make_dataset(TestImgPath)) != int(
            cv2.VideoCapture(VideoPath).get(cv2.CAP_PROP_FRAME_COUNT)
        ):
            # Extract frames to input folder.
            print("\nExtracting video frames...\n")
            args = [
                "ffmpeg",
                "-i",
                "{}".format(VideoPath),
                "{}\\%d.png".format(SaveInputPath),
            ]
            subprocess.run(args)

    #######################################################################
    ###########Step 1: Crop and Align Face from the whole Image ###########
    #######################################################################
    print(
        "\n###############################################################################"
    )
    print(
        "####################### Step 1: Crop and Align Face ###########################"
    )
    print(
        "###############################################################################\n"
    )
    detector = dlib.cnn_face_detection_model_v1(
        "./packages/mmod_human_face_detector.dat"
    )
    sp = dlib.shape_predictor("./packages/shape_predictor_5_face_landmarks.dat")
    reference = np.load("./packages/FFHQ_template.npy") / 2
    ImgPaths = [
        ImgPath
        for ImgPath in make_dataset(TestImgPath)
        if not os.path.exists(
            os.path.join(SaveParamPath, os.path.split(ImgPath)[-1] + ".npy")
        )
    ]
    for ImgPath in tqdm(ImgPaths):
        for _ in range(2):
            ImgName = os.path.split(ImgPath)[-1]
            SavePath = os.path.join(SaveCropPath, ImgName)
            SaveInput = os.path.join(SaveInputPath, ImgName)
            SaveParam = os.path.join(
                SaveParamPath, ImgName + ".npy"
            )  # Save the inverse affine parameters.
            try:
                align_and_save(
                    ImgPath,
                    SavePath,
                    None if VideoPath else SaveInput,
                    SaveParam,
                    UpScaleWhole,
                )
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    exit()
                elif "cudaGetLastError()" not in str(e):
                    print(
                        "\n\t################ Error in extracting from this image: {}".format(
                            ImgName
                        )
                    )
                    break

    #######################################################################
    ####### Step 2: Face Landmark Detection from the Cropped Image ########
    #######################################################################
    print(
        "\n###############################################################################"
    )
    print(
        "####################### Step 2: Face Landmark Detection #######################"
    )
    print(
        "###############################################################################\n"
    )
    if len(opt.gpu_ids) > 0:
        dev = "cuda:{}".format(opt.gpu_ids[0])
    else:
        dev = "cpu"
    FD = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, device=dev, flip_input=False
    )
    ImgPaths = [
        ImgPath
        for ImgPath in make_dataset(SaveCropPath)
        if not os.path.exists(
            os.path.join(SaveLandmarkPath, os.path.split(ImgPath)[-1] + ".txt")
        )
    ]
    for i, ImgPath in enumerate(tqdm(ImgPaths)):
        ImgName = os.path.split(ImgPath)[-1]
        Img = io.imread(ImgPath)
        try:
            PredsAll = FD.get_landmarks(Img)
        except:
            print("\n\t################ Error in face detection, continue...")
            continue
        if PredsAll is None:
            print("\n\t################ No face, continue...")
            continue
        ins = 0
        if len(PredsAll) != 1:
            hights = []
            for l in PredsAll:
                hights.append(l[8, 1] - l[19, 1])
            ins = hights.index(max(hights))
        preds = PredsAll[ins]
        AddLength = np.sqrt(np.sum(np.power(preds[27][0:2] - preds[33][0:2], 2)))
        SaveName = ImgName + ".txt"
        np.savetxt(os.path.join(SaveLandmarkPath, SaveName), preds[:, 0:2], fmt="%.3f")

    #######################################################################
    ####################### Step 3: Face Restoration ######################
    #######################################################################
    print(
        "\n###############################################################################"
    )
    print(
        "####################### Step 3: Face Restoration ##############################"
    )
    print(
        "###############################################################################\n"
    )
    print(f"Loading {UpScaleWhole}x model...")
    start_time: float = time.time()
    model = create_model(opt)
    model.setup(opt)
    print("Loaded model, took {:.2f} seconds.\n".format(time.time() - start_time))
    ImgPaths = [
        ImgPath
        for ImgPath in make_dataset(SaveCropPath)
        if not os.path.exists(os.path.join(SaveRestorePath, os.path.split(ImgPath)[-1]))
    ]
    total = 0
    for i, ImgPath in enumerate(tqdm(ImgPaths)):
        ImgName = os.path.split(ImgPath)[-1]
        torch.cuda.empty_cache()
        data = obtain_inputs(SaveCropPath, SaveLandmarkPath, ImgName)
        if data == 0:
            print("\n\t################ Error in landmark file, continue...")
            continue
        total = total + 1
        model.set_input(data)
        try:
            model.test()
            visuals = model.get_current_visuals()
            save_crop(visuals, os.path.join(SaveRestorePath, ImgName))
        except:
            print(
                "\n\t################ Error in enhancing this image: {}".format(ImgName)
            )
            print("\t################ continue...")
            continue

    #######################################################################
    ############ Step 4: Paste the Results to the Input Image #############
    #######################################################################
    print(
        "\n###############################################################################"
    )
    print(
        "############### Step 4: Paste the Restored Face to the Input Image ############"
    )
    print(
        "###############################################################################\n"
    )

    ImgPaths = [
        ImgPath
        for ImgPath in make_dataset(SaveRestorePath)
        if not os.path.exists(os.path.join(SaveFinalPath, os.path.split(ImgPath)[-1]))
    ]
    Threads: int = max(1, min(len(ImgPaths), int(os.cpu_count() / UpScaleWhole)))
    print("Using {} thread(s)...\n".format(Threads))
    pool: multiprocessing.Pool = multiprocessing.Pool(Threads)
    try:
        Params = [
            (
                TestImgPath,
                SaveRestorePath,
                SaveParamPath,
                SaveFinalPath,
                UpScaleWhole,
                ImgPath,
            )
            for ImgPath in ImgPaths
        ]
        for _ in tqdm(pool.imap_unordered(worker, Params), total=len(ImgPaths)):
            pass
    except KeyboardInterrupt:
        exit(1)

    # Video output.
    if VideoPath:
        print("\nCombining final frames with input frames...\n")
        FramesPath = os.path.join(ResultsDir, "TemporaryFrames")
        if not os.path.exists(FramesPath):
            os.makedirs(FramesPath)
        FramesPathNames = {
            os.path.basename(ImgPath): ImgPath
            for ImgPath in make_dataset(SaveInputPath)
        }
        for ImgPath in make_dataset(SaveFinalPath):
            FramesPathNames[os.path.basename(ImgPath)] = ImgPath
        for FramePathName, FramePath in tqdm(FramesPathNames.items()):
            shutil.copyfile(FramePath, os.path.join(FramesPath, FramePathName))

        # Make video.
        print("\nCreating video...\n")
        args = [
            "ffmpeg",
            "-i",
            "{}".format(VideoPath),
            "-framerate",
            str(cv2.VideoCapture(VideoPath).get(cv2.CAP_PROP_FPS)),
            "-i",
            "{}\\%d.png".format(FramesPath),
            "-map",
            "0:a",
            "-map",
            "1:v",
            "-y",
            "{}\\{}".format(ResultsDir, os.path.split(VideoPath)[-1]),
        ]
        subprocess.run(args)

    print("\nAll results are saved in {}".format(ResultsDir))

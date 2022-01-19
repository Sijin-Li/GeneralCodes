import cv2

'''
binary segmentation
according to a user-defined threshold
'''

def shadowextractor(inputdata, savepath, filename, inverse=False, threshold=50):

    # shadow == 1
    os.makedirs(savePath, exist_ok=True)
    savefilePath = os.path.join(savepath, filename)

    HSV_img = cv2.cvtColor(inputdata, cv2.COLOR_BGR2HSV)
    Hchannel = HSV_img[:, :, 0]
    retShadow, Shadowimg = cv2.threshold(Hchannel, threshold, 255, cv2.THRESH_BINARY_INV)

    cv2.imwrite(savefilePath, Shadowimg)

    # inverse shadow
    # shadow == 0
    if inverse is True:
        inverse_savePath = savePath.replace('shadow_crop', 'invshadow_crop')
        os.makedirs(inverse_savePath, exist_ok=True)
        retInverse, Inverseimg = cv2.threshold(Hchannel, threshold, 255, cv2.THRESH_BINARY)

        inverse_savefilePath = os.path.join(inverse_savePath, filename)
        cv2.imwrite(inverse_savefilePath, Inverseimg)

    print("done")

